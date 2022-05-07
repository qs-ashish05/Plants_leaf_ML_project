import filecmp
from multiprocessing import connection
from pdb import main
from re import U
import sqlite3
from tkinter.filedialog import askopenfilename
import numpy as np
import os
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


from io import BytesIO
from PIL import Image

from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox

print("dependencies are imported")
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

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = ds.cardinality().numpy()
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  seed=123,
  image_size=default_image_size,
  batch_size=BATCH_SIZE
)

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)



def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


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



def GUI_system():
    
    root = Tk()

    root.geometry('1000x250')
    root.title("Sem-IV mini project")
    root.configure(bg="#a786f4")

    Canvas = Canvas(width=400,height=250,bg="#a786f4")
    Canvas.pack()

    photo = PhotoImage(file = 'photo()')
    Canvas.Create_Image(0,0,image=photo,anchor= NW)


    root.mainloop()

def OpenFile():
    
    if y:
        try:
            a = askopenfilename()
            print(a)
            value, classes = main()
            messagebox.showinfo("your report", ("Predicted class is  ", "\nPredicted Class is ", classes))

            query = 'UPDATE THEGREAT SET PREDICT = "%s" WHERE USERNAME = "%s"'%(value, username)

            sqlite3.execute(query)
            #print(query)
            connection.commit()

            #------********************Only use when required to send message
            #send(value, classes)
            #------*********************************************************
            image = Image.open(a)
            # plotting image
            file = image.convert('RGB')
            plt.imshow(np.array(file))
            plt.title(f'your report is label : {value} class : {classes}')
            plt.show()
            #print(image)
            print('Thanks for using the system !')
           
        except Exception as error:
           GUI_system()



x = 0
y = True


print("GUI system started")
root = Tk()

root.geometry('1000x250')
root.title("Group 14")
root.configure(bg='#a786f4')


label1 = Label(root, text="Leaf disease detection using machine learning ", font=('Arial', 30))
label1.grid(padx=30, pady=30, row=0, column=0, sticky='W')

label2 = Label(root, text="DREAM", font=('helvetica', 20))
label2.grid(padx=30, pady=10, row=1, column=0, sticky='W')





button2 = Button(root, text="Upload Image", command=OpenFile)
button2.grid(padx=10, pady=20, row=2, column=0)


root.mainloop()