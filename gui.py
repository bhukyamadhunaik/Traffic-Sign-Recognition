# gui.py

import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Load trained model

model = load_model('traffic_classifier.h5')

# Dictionary of classes

classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing veh > 3.5 tons'
}

# GUI Setup

top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('Arial', 15, 'bold'))
sign_image = Label(top)

# Functions

def classify(file_path):
    try:
        img = Image.open(file_path).resize((30, 30))
        img_arr = np.expand_dims(np.array(img), axis=0)
        pred_probs = model.predict(img_arr)[0]
        pred_class = np.argmax(pred_probs) + 1  # offset to match dict keys
        sign_text = classes.get(pred_class, "Unknown")
        label.configure(foreground='#011638', text=sign_text)
    except Exception as e:
        label.configure(text=f"Error: {e}")

def show_classify_button(file_path):
    classify_btn = Button(top, text="Classify Image",
                          command=lambda: classify(file_path),
                          background='#364156', foreground='white',
                          font=('Arial', 10, 'bold'))
    classify_btn.place(relx=0.79, rely=0.46)

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((top.winfo_width()/2.25, top.winfo_height()/2.25))
        img_tk = ImageTk.PhotoImage(img)
        sign_image.configure(image=img_tk)
        sign_image.image = img_tk
        label.configure(text='')
        show_classify_button(file_path)

# GUI Layout

upload_btn = Button(top, text="Upload an image", command=upload_image,
                    background='#364156', foreground='white',
                    font=('Arial', 10, 'bold'))
upload_btn.pack(side=tk.BOTTOM, pady=50)

sign_image.pack(side=tk.BOTTOM, expand=True)
label.pack(side=tk.BOTTOM, expand=True)

heading = Label(top, text="Know Your Traffic Sign", pady=20,
                font=('Arial', 20, 'bold'),
                background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()

