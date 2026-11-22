// Use your computer's local IP address (same Wi-Fi as your phone), NOT "localhost".
// Android emulator: http://10.0.2.2:8000   |   iOS simulator: http://localhost:8000
export const API_URL = "http://192.168.0.105:8000";

// `key` must match the keys in backend/main.py (MODEL_FILES)
// `size` is the input size the model expects; the app resizes the photo to this before upload
export const MODELS = [
  { key: "cnn", name: "CNN", accuracy: 70, size: 256 },
  { key: "vgg16", name: "VGG16", accuracy: 90, size: 128 },
  { key: "inception", name: "Inception V1", accuracy: 93, size: 224 },
  { key: "vit", name: "Vision Transformer", accuracy: 90, size: 128 },
];
