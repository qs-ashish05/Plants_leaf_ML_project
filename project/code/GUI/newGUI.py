import filecmp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
from io import BytesIO
from PIL import Image
from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog

print("Dependencies are loaded")

data_dir = "PlantVillage"
IMAGE_SIZE = 256
BATCH_SIZE = 32
default_image_size = tuple((IMAGE_SIZE, IMAGE_SIZE))

# load model 
MODEL = tf.keras.models.load_model("../models/1")
print("model loaded")

CLASS_NAMES = ['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_healthy']


print("GUI system started")

root = Tk()
root.title('Sem 4 project')


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

def photo():
    plt.figure(figsize=(5, 5))
    for images, labels in test_ds.take(50):
        for i in range(1):
            ax = plt.subplot(1, 1, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_class, confidence = predict(MODEL, images[i].numpy())
            
            actual_class = class_names[labels[i]] 
            plt.title(f"Actual Image: {actual_class},\n Predicted Image: {predicted_class}.\n Accuracy: {confidence}%")
            img = plt.axis("off")
            return img

plt.figure(figsize=(4, 4))
for images, labels in test_ds.take(15):
    for i in range(1):
        ax = plt.subplot(1, 1, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(MODEL, images[i].numpy())
        
        actual_class = class_names[labels[i]] 
        plt.title(f"Actual Image: {actual_class},\n Predicted Image: {predicted_class}.\n Accuracy: {confidence}%")
        plt.axis("off")

def open():
    global img
    root.filename = filedialog.askopenfilename(initialdir="F:\project\code\GUI\Sample",title="Select a file",filetypes=(("png files",".png"),("all files","*.*")))
    img = ImageTk.PhotoImage(Image.open(root.filename))
    img_label = Label(image=img).pack()

btn = Button(root, text="open file",command=open).pack()
    
root.mainloop()
# root = Tk()

# root.geometry('1000x250')
# root.title("Mini project")
# root.configure(bg='#a786f4')

# def open():
#     global img
#     root.filename = filedialog.askopenfilename(initialdir="F:\project\code\GUI\Sample",title="Select a file",filetypes=(("png files",".png"),("all files","*.*")))
#     img = ImageTk.PhotoImage(Image.open(root.filename))
#     img_label = Label(image=img)
#     label1.grid(padx=30, pady=30, row=0, column=0, sticky='W')

# label1 = Label(root, text="Leaf disease detection using CNN", font=('Arial', 30))
# label1.grid(padx=30, pady=30, row=0, column=0, sticky='W')

# label2 = Label(root, text="DREAM", font=('helvetica', 20))
# label2.grid(padx=30, pady=10, row=1, column=0, sticky='W')

# button2 = Button(root, text="Upload Image", command=open)
# button2.grid(padx=10, pady=20, row=2, column=0)


# root.mainloop()