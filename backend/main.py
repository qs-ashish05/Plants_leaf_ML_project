"""
Inference server for the potato-leaf disease app.

Loads the four trained models (CNN .h5, VGG16 / Inception V1 / ViT .pth) and
exposes a single /predict endpoint that the React Native app calls.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models

# ----------------------------------------------------------------------------
# Config - edit these to match your files
# ----------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "models"

MODEL_FILES = {
    "cnn": "cnnArch.h5",
    "vgg16": "vgg16.pth",
    "inception": "inception.pth",
    "vit": "vision_transformer.pth",
}

# Image size each model was trained with (used if the checkpoint doesn't store it)
DEFAULT_SIZES = {"cnn": 256, "vgg16": 128, "inception": 224, "vit": 128}

# Alphabetical folder order of PlantVillage potato classes (what ImageFolder gives).
# Used for the CNN (.h5 has no class names) and for any checkpoint without "classes".
DEFAULT_CLASSES = ["Potato___Early_blight",
                   "Potato___Late_blight", "Potato___healthy"]

# Inception was fine-tuned from ImageNet weights (PRETRAINED = True in the training script).
# Keep True unless you trained it from scratch.
INCEPTION_PRETRAINED = True

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------
# Tiny ViT - must match the architecture used in training exactly
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, C //
                                   self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, n_classes, img_size=128, patch=16, dim=192, depth=6,
                 heads=3, mlp_ratio=2, dropout=0.1):
        super().__init__()
        n_tokens = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = self.drop(x + self.pos_embed)
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])


# ----------------------------------------------------------------------------
# Model loaders. Each returns: dict(kind, model, classes, size, norm)
# ----------------------------------------------------------------------------
def _read_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt
    return ckpt, {}


def load_vgg16(path):
    state, meta = _read_checkpoint(path)
    classes = meta.get("classes", DEFAULT_CLASSES)
    model = nn.Sequential(
        models.vgg16(weights=None).features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, len(classes)),
    )
    model.load_state_dict(state)
    return dict(kind="torch", model=model.eval(), classes=classes,
                size=meta.get("image_size", DEFAULT_SIZES["vgg16"]), norm="imagenet")


def load_inception(path):
    state, meta = _read_checkpoint(path)
    classes = meta.get("classes", DEFAULT_CLASSES)
    model = models.googlenet(weights=None, aux_logits=False, init_weights=False,
                             transform_input=INCEPTION_PRETRAINED, num_classes=len(classes))
    model.load_state_dict(state)
    return dict(kind="torch", model=model.eval(), classes=classes,
                size=meta.get("image_size", DEFAULT_SIZES["inception"]), norm="imagenet")


def load_vit(path):
    state, meta = _read_checkpoint(path)
    classes = meta.get("classes", DEFAULT_CLASSES)
    size = meta.get("image_size", DEFAULT_SIZES["vit"])
    if meta.get("pretrained_timm", False):
        import timm
        model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False,
                                  img_size=size, num_classes=len(classes))
    else:
        model = TinyViT(len(classes), img_size=size)
    model.load_state_dict(state)
    return dict(kind="torch", model=model.eval(), classes=classes, size=size, norm="half")


def load_cnn(path):
    # imported lazily so the server starts without TF if you don't need the CNN
    import tensorflow as tf
    model = tf.keras.models.load_model(str(path), compile=False)
    return dict(kind="keras", model=model, classes=DEFAULT_CLASSES,
                size=DEFAULT_SIZES["cnn"], norm="raw")


LOADERS = {"cnn": load_cnn, "vgg16": load_vgg16,
           "inception": load_inception, "vit": load_vit}

_cache = {}
_lock = threading.Lock()


def get_model(key):
    """Load a model the first time it is requested, then keep it in memory."""
    with _lock:
        if key not in _cache:
            path = MODEL_DIR / MODEL_FILES[key]
            if not path.exists():
                raise HTTPException(404, f"Model file not found: {path}")
            print(f"Loading {key} from {path} ...")
            _cache[key] = LOADERS[key](path)
        return _cache[key]


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def run_model(entry, img):
    size = entry["size"]
    img = img.resize((size, size), Image.BILINEAR)
    # (H, W, 3), 0-255
    arr = np.asarray(img, dtype=np.float32)

    if entry["kind"] == "keras":
        # The notebook model has Resizing + Rescaling layers inside, so it expects raw 0-255 pixels
        probs = entry["model"].predict(arr[None], verbose=0)[0]
        if not np.isclose(probs.sum(), 1.0, atol=1e-3):          # in case it outputs logits
            e = np.exp(probs - probs.max())
            probs = e / e.sum()
        return probs

    x = torch.from_numpy(arr / 255.0).permute(2, 0, 1).unsqueeze(0)
    if entry["norm"] == "imagenet":
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    else:
        x = (x - 0.5) / 0.5
    with torch.inference_mode():
        return torch.softmax(entry["model"](x), dim=1)[0].numpy()


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
app = FastAPI(title="Potato Leaf Disease API")
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODEL_FILES)}


class Base64Request(BaseModel):
    # base64-encoded JPEG/PNG (a "data:...;base64," prefix is allowed)
    image: str
    model: str


@app.post("/predict")
def predict(file: UploadFile = File(...), model: str = Form(...)):
    """Multipart upload version (e.g. for testing with curl or Postman)."""
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Uploaded file is not a valid image")
    return _predict_image(model, img)


@app.post("/predict_base64")
def predict_base64(req: Base64Request):
    """JSON version used by the mobile app."""
    data = req.image.split(",", 1)[1] if req.image.startswith(
        "data:") else req.image
    try:
        img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode the uploaded image")
    return _predict_image(req.model, img)


def _predict_image(model, img):
    if model not in MODEL_FILES:
        raise HTTPException(
            400, f"Unknown model '{model}'. Choose from {list(MODEL_FILES)}")

    entry = get_model(model)
    probs = run_model(entry, img)

    classes = entry["classes"]
    names = classes if len(classes) == len(probs) else [
        f"class_{i}" for i in range(len(probs))]
    best = int(np.argmax(probs))

    return {
        "model": model,
        "label": names[best],
        "confidence": round(float(probs[best]) * 100, 2),
        "probabilities": {n: round(float(p) * 100, 2) for n, p in zip(names, probs)},
        "input_size": entry["size"],
    }
