from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


MODEL = tf.keras.models.load_model("../model/1")

CLASS_NAMES = ['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_healthy']

def get_infection(prediction):
    if prediction == "Pepper__bell___Bacterial_spot":
        I = "Papper Bell Bacterial Spot"
        return I

    elif  prediction == "Potato___Early_blight":
        I = "Potato Early Blight"
        return I

    elif  prediction == "Potato___Late_blight":
        I = "Potato Late Blight"
        return I

    elif  prediction == "Tomato_Early_blight":
        I = "Tomato Early Blight"
        return I

    elif  prediction == "Tomato_Late_blight":
        I = "Tomato Late Blight"
        return I

    elif  prediction == "Pepper__bell___healthy":
        I = "Pepper Bell Healthy"
        return I
    elif  prediction =="Potato___healthy":
        I = "Potato Healty"
        return I
    elif  prediction == "Tomato_healthy":
        I = "Tomato Healthy"
        return I


def get_prev(prediction):
    if prediction == "Pepper__bell___Bacterial_spot":
        P = "(1) Seed treatment with hot water, soaking seeds for 30 minutes in water pre-heated to 125 F/51 C. \n (2)Use a copper based fungicide as a foliar spray in the early morning or late evening to help reduce the spread"
        return P

    elif  prediction == "Potato___Early_blight":
        P = "(1) Treatment of early blight includes prevention by planting potato varieties that are resistant to the disease; late maturing are more resistant than early maturing varieties. \n(2) Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible."
        return P

    elif  prediction == "Potato___Late_blight":
        P = "(1) Late blight is controlled by eliminating cull piles and volunteer potatoes \n (2) Using proper harvesting and storage practices \n (3) applying fungicides when necessary."
        return P

    elif  prediction == "Tomato_Early_blight":
        P = "(1) Use Liquid Copper Fungicide \n (2) Spray the plant with liquid copper fungicide concentrate"
        return P

    elif  prediction == "Tomato_Late_blight":
        P = "(1) copper fungicide can be applied and used to treat late blight effectively. \n(2) Water Properly \n(3) Pull out the plant from the garden as soon as possible"
        return P

    elif  prediction == "Pepper__bell___healthy" or "Potato___healthy" or "Tomato_healthy":
        P = "NILL"
        return P



def get_cause(prediction):
    if prediction == "Pepper__bell___Bacterial_spot":
        cause = "Xanthomonas campestris pv. vesicatoria"
        return cause

    elif  prediction == "Potato___Early_blight":
        cause = "Fungus-Alternaria solani"
        return cause

    elif  prediction == "Potato___Late_blight":
        cause = "Fungus-Phytophthora infestans"
        return cause

    elif  prediction == "Tomato_Early_blight":
        cause = "Fungus-Alternaria solani"
        return cause

    elif  prediction == "Tomato_Late_blight":
        cause = "Fungus-Phytophthora infestans"
        return cause

    elif  prediction == "Pepper__bell___healthy" or "Potato___healthy" or "Tomato_healthy":
        cause = "NILL"
        return cause


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    prediction = CLASS_NAMES[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])
    confidence = float(confidence)
    infection = get_infection(prediction)
    cause = get_cause(prediction)
    prevention = get_prev(prediction)

        

    return {
        'Infection': infection,
        'Accuarcy': float(confidence),
        "Cause": cause,
        "prevention" : prevention
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

