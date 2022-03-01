from distutils.command.upload import upload
from msilib.schema import File
from fastapi import FastAPI, UploadFile,File
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


# MODE = tf.keras.models.load_model("E:\FINAL_SEM_IV_PROJECTS\Plants_leaf_ML_project\Codes\model\1")
MODEL = tf.keras.models.load_model("..\model\1")
CLASS_NAMES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy']




@app.get("/test")
async def test():
    return "Server is Running"


def read_file_as_image(data) -> np.ndarray:
   image =  np.array(Image.opem(BytesIO(data)))
   return image


@app.post("/predict")
async def predict(

    file:UploadFile = File(...)  


):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    index = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[index]
    confidence = np.max(prediction[0])

    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":

    uvicorn.run(app, host ='localhost', port =8000)