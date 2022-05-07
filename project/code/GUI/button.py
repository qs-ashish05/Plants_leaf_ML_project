from logging import root
from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog

root = Tk()
root.title('Sem 4 project')

def open():
    
    root.geometry('200x150')
    global img
    root.filename = filedialog.askopenfilename(initialdir="F:\project\code\GUI\Sample",title="Select a file",filetypes=(("png files",".png"),("all files","*.*")))
    my_label = Label(root,text="uploaded image")
    img = ImageTk.PhotoImage(Image.open(root.filename))
    img_label = Label(image=img)

btn = Button(root, text="open file",command=open).pack()
    
root.mainloop()