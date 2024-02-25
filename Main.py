import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import cv2 as cv
from PIL import Image, ImageTk, ImageOps, ImageGrab

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torchsummary import summary

import random

area = (400, 200)
area2 = 80000
img2_label = None  # Initialize img2_label as a global variable

# Q1
def load_video():
    global path
    path = filedialog.askopenfilename()
    if path:
        print('Selected video:', path)

# Q3 
def load_img():
    global img_path
    img_path = filedialog.askopenfilename()
    if img_path:
        print('Selected image: ', img_path)


# Q4
def model1():
    model_vgg = models.vgg19_bn(num_classes=10, weights=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_vgg.load_state_dict(torch.load('model/Cv_Dl_HW2_model1.pth', 
                                 map_location=device))
    summary(model_vgg, (3, 32, 32))
#     col_names=['output_size', 'num_params'])

def results1():
    global img_label
    # Open an image file
    img = Image.open('VGG19.png')
    img = img.resize(area)
    # Display the image
    res_1_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(canvas, image=res_1_tk)
    img_label.image = res_1_tk
    img_label.grid(row=0)

def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    thickness = 10
    global last_x, last_y
    # canvas.create_oval(last_x, last_y, event.x, event.y, fill="white", width=thickness)
    # canvas.create_oval(last_x+1, last_y+1, event.x, event.y, fill="white", width=thickness)
    # canvas.create_oval(last_x-1, last_y-1, event.x, event.y, fill="white", width=thickness)
    canvas.create_line(last_x, last_y, event.x, event.y,
                            width=thickness, fill="white")
    last_x, last_y = event.x, event.y



def reset_canvas():
    global img_label
    canvas.delete('all')
    if img_label is not None:  # If the label exists, destroy it
        img_label.destroy()
        img_label = None  # Reset the reference

def inference():
    global pred_image, prediction_label

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    model_vgg = models.vgg19_bn(num_classes=10, weights=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_vgg.load_state_dict(torch.load('model/Cv_Dl_HW2_model1.pth', 
                                 map_location=device))

    model_vgg.eval()

    # Transform
    transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the image
    # canvas.postscript(file='/Users/davidhernandez/Desktop/NCKU/Computer_Vision/HW2/drawing.eps', colormode='mono')
    # # Save the drawn image as a grayscale image with a black background
    # img = Image.open('/Users/davidhernandez/Desktop/NCKU/Computer_Vision/HW2/drawing.eps')
    # img.show()
    # img = ImageOps.grayscale(img)
    # img = img.point(lambda p: p > 128 and 255)

    # # Save as a different file type (e.g., PNG)
    # img.save('/Users/davidhernandez/Desktop/NCKU/Computer_Vision/HW2/testing.png')
    # img = Image.open('/Users/davidhernandez/Desktop/NCKU/Computer_Vision/HW2/testing.png')

    canvas_x1, canvas_y1 = canvas.winfo_rootx(), canvas.winfo_rooty()
    canvas_x2, canvas_y2 = canvas_x1 + canvas.winfo_width(), canvas_y1 + canvas.winfo_height()
    canvas_image = ImageGrab.grab((canvas_x1, canvas_y1, canvas_x2, canvas_y2))
    canvas_image.save("canvas_image.png")  # Save the captured image to a file
    img = transform_test(canvas_image).unsqueeze(0)
    print(img)
    # Make a prediction
    with torch.no_grad():
        output = model_vgg(img)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities)
        print(predicted_class)

    # Display the prediction
    prediction_text = f"Predicted: {classes[predicted_class.item()]}"
    prediction_label = tk.Label(pic_frame_1, text=prediction_text)
    prediction_label.grid(row=3)
    print(predicted_class.item())

    probabilities_np = probabilities.numpy()[0]

    # Create a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(10), probabilities_np)
    plt.xticks(np.arange(10), classes)
    plt.title('Probability Distribution of Model Predictions')
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.show()


# Q5
def load_imgs():
    global imgs_path, img2_label
    reset_canvas2()
    imgs_path = filedialog.askopenfilename()
    if imgs_path:
        print('Selected image: ', imgs_path)
    img = Image.open(imgs_path)
    area_image = img.size[0] * img.size[1]
    if area_image > area2:
    # Calculate the scaling factor to fit within area2
        scaling_factor = (area2 / area_image) ** 0.5
        new_width = int(img.size[0] * scaling_factor)
        new_height = int(img.size[1] * scaling_factor)
        image = img.resize((new_width, new_height))
    else:
        image = img

    res_2_tk = ImageTk.PhotoImage(image)
    img2_label = tk.Label(canvas2, image=res_2_tk)
    img2_label.image = res_2_tk
    img2_label.grid(row=0)

def show_img():
    cat_folder = "inference_dataset/Cat"
    dog_folder = "inference_dataset/Dog"

    cat_files = os.listdir(cat_folder)
    dog_files = os.listdir(dog_folder)

    cat_files = [file for file in cat_files if file.endswith((".jpg", ".png", ".jpeg"))]
    dog_files = [file for file in dog_files if file.endswith((".jpg", ".png", ".jpeg"))]

    random_cat = random.choice(cat_files)
    random_dog = random.choice(dog_files)

    cat = plt.imread(os.path.join(cat_folder, random_cat))
    dog = plt.imread(os.path.join(dog_folder, random_dog))


    fig, ax = plt.subplots(1,2,figsize = (5,5))
    ax = ax.flatten()
    ax[0].imshow(cat)
    ax[0].set_axis_off()
    ax[0].set_title('Cat', fontsize=12, y=1.05, ha="center")
    ax[1].imshow(dog)
    ax[1].set_axis_off()
    ax[1].set_title("Dog", fontsize=12, y=1.05, ha="center")

    plt.tight_layout()
    plt.show()

def model2():
    model_rn = models.resnet50(weights=None)
    #add a new final layer
    model_rn.fc = nn.Sequential(
    nn.Linear(2048, 1),
    nn.Sigmoid()
    )
    summary(model_rn, (3, 224, 224))
    # col_names=['output_size', 'num_params'])

def comparison():
    # Data
    categories = ['Without Random erasing', 'With Random erasing']
    values = [96.44, 97.17]
    y_axis_range = 100

    # Create the bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(categories, values, width=0.5, align='center')

    # Set the y-axis range
    ax.set_ylim(0, y_axis_range)

    # Add labels and title
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison')

    # Add data values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval/2, round(yval, 2), ha='center', va='center')

    return fig, ax

def show_comparison():
    fig, ax = comparison()
    plt.show()

def inference2():
    global pred_image, prediction_label

    classes = ('Cat', 'Dog')

    model_rn = models.resnet50(weights=None)
    #add a new final layer
    model_rn.fc = nn.Sequential(
    nn.Linear(2048, 1),
    nn.Sigmoid()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_rn.load_state_dict(torch.load('model/Cv_Dl_ResNet_v2.pth', 
                                 map_location=device))

    model_rn.eval()

    test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],),
    ])

    image = Image.open(imgs_path)
    img = test_transform(image).unsqueeze(0)
    # Make a prediction
    with torch.no_grad():
        output = model_rn(img)
        probabilities = torch.nn.functional.sigmoid(output)  # Use sigmoid activation
        predicted_class = 1 if probabilities > 0.5 else 0  # Interpret based on sigmoid output
        print(predicted_class)

    # Display the prediction
    predicted_class_label = 'Cat' if predicted_class == 1 else 'Dog'
    prediction_text = f"Predicted: {predicted_class_label}"
    prediction_label = tk.Label(pic_frame_2, text=prediction_text)
    prediction_label.grid(row=3)

    probabilities_np = probabilities.numpy()[0]


def reset_canvas2():
    global img2_label
    canvas2.delete('all')
    if img2_label is not None:  # If the label exists, destroy it
        img2_label.destroy()
        img2_label = None  # Reset the reference

# Create the main window
root = tk.Tk()
root.title("Classification VGG19 and ResNet50")

# Create frame for loading stuff
left_frame = tk.Frame(root)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')

# Create buttons and labels
load_image = tk.Button(left_frame, text='Load Image', command=load_img)
load_image.grid(row=0, padx=10, pady=10, sticky='w')

load_vid = tk.Button(left_frame, text='Load Video', command=load_video)
load_vid.grid(row=1, padx=10, pady=10, sticky='w')


# Frame for Q4
frame_4 = tk.Frame(root)
frame_4.grid(row=0, column=2, padx=10, pady=10, sticky='w')

# Frame for Q4's Image
pic_frame_1 = tk.Frame(root)
pic_frame_1.grid(row=0, column=3, padx=10, pady=10, sticky='w')
# Frame for Q5
frame_5 = tk.Frame(root)
frame_5.grid(row=2, column=2, padx=10, pady=10, sticky='w')

# Frame for Q5's Image
pic_frame_2 = tk.Frame(root)
pic_frame_2.grid(row=2, column=3, padx=10, pady=10, sticky='w')



#Label 4 
Q4_label = tk.Label(frame_4, text='4. MNIST Classifier Using VGG19')
Q4_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

# Add a canvas for drawing
canvas = tk.Canvas(pic_frame_1, width=area[0], height=area[1], bg='black')
canvas.grid(row=0, column=0, padx=10, pady=10)
canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
# Q4 Buttons
Model_1 = tk.Button(frame_4,text='1. Show Model Structure', command=model1)
Model_1.grid(row=1)

Performance_1 = tk.Button(frame_4,text='2. Show Accuracy and Loss', command=results1)
Performance_1.grid(row=2)

Predict_1 = tk.Button(frame_4, text='3. Predict', command=inference)
Predict_1.grid(row=3)

Reset = tk.Button(frame_4, text='4. Reset', command=reset_canvas)
Reset.grid(row=4)


#Label 5
Q5_label = tk.Label(frame_5, text='5. ResNet50')
Q5_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

canvas2 = tk.Canvas(pic_frame_2, width=area[0], height=area[1], bg='black')
canvas2.grid(row=0, column=0, padx=10, pady=10)

# Q5 Buttons
ld_img = tk.Button(frame_5, text='Load Image', command=load_imgs)
ld_img.grid(row=1)

show_img = tk.Button(frame_5, text='5.1 Show Images', command=show_img)
show_img.grid(row=2)

Model_2 = tk.Button(frame_5, text='5.2 Show Model Structure', command=model2)
Model_2.grid(row=3)

Performance_2 = tk.Button(frame_5, text='5.3 Show Comparison', command=show_comparison)
Performance_2.grid(row=4)

infer = tk.Button(frame_5, text='5.4 Inference', command=inference2)
infer.grid(row=5)



# Start the GUI event loop
root.mainloop()