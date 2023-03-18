import tkinter as tk
from PIL import Image, ImageDraw2, ImageOps
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tkinter import *
from PIL import Image, ImageDraw
from sklearn import svm
import joblib
import io

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define the threshold value
threshold_value = 127

# Threshold the images
x_train_thresholded = np.where(x_train > threshold_value, 1, 0)
x_test_thresholded = np.where(x_test > threshold_value, 1, 0)

### BOUNDING BOX
def construct_bounding_box(image):
    # Compute row-wise and column-wise sums of the thresholded image
    row_sums = np.sum(image, axis=1)
    col_sums = np.sum(image, axis=0)

    # Find the range of ink pixels along each row and column
    row_nonzero = np.nonzero(row_sums)[0]
    col_nonzero = np.nonzero(col_sums)[0]
    if len(row_nonzero) == 0 or len(col_nonzero) == 0:
        return np.zeros((20, 20))
    row_range = row_nonzero[[0, -1]]
    col_range = col_nonzero[[0, -1]]

    # Compute the center of the ink pixel ranges
    row_center = (row_range[0] + row_range[-1]) / 2
    col_center = (col_range[0] + col_range[-1]) / 2

    # Compute starting and ending indices for the bounding box
    row_start = int(np.clip(row_center - 9, 0, image.shape[0] - 20))
    row_end = row_start + 20
    col_start = int(np.clip(col_center - 9, 0, image.shape[1] - 20))
    col_end = col_start + 20

    # Extract the bounding box from the image
    bounding_box = image[row_start:row_end, col_start:col_end]

    return bounding_box

# Streched box bounding
def construct_bounding_box_stretched(image):
    # Compute row-wise and column-wise sums of the thresholded image
    row_sums = np.sum(image, axis=1)
    col_sums = np.sum(image, axis=0)

    # Find the range of ink pixels along each row and column
    row_nonzero = np.nonzero(row_sums)[0]
    col_nonzero = np.nonzero(col_sums)[0]
    if len(row_nonzero) == 0 or len(col_nonzero) == 0:
        return np.zeros((20, 20))

    # Compute the horizontal and vertical ink pixel ranges
    row_range = row_nonzero[[0, -1]]
    col_range = col_nonzero[[0, -1]]
    row_start, row_end = row_range[0], row_range[-1]
    col_start, col_end = col_range[0], col_range[-1]

    # Stretch the extracted image to 20x20 dimensions
    image = image[row_start:row_end, col_start:col_end]
    image = resize(image, (20, 20))

    return image


x_train_bounding_box = np.zeros((len(x_train_thresholded), 20, 20))
x_train_bounding_box_stretched = np.zeros((len(x_train_thresholded), 20, 20))
for i in range(len(x_train_thresholded)):
    x_train_bounding_box[i] = construct_bounding_box(x_train_thresholded[i])
    x_train_bounding_box_stretched[i] = construct_bounding_box_stretched(x_train_thresholded[i])

x_test_bounding_box = np.zeros((len(x_test_thresholded), 20, 20))
x_test_bounding_box_stretched = np.zeros((len(x_test_thresholded), 20, 20))
for i in range(len(x_test_thresholded)):
    x_test_bounding_box[i] = construct_bounding_box(x_test_thresholded[i])
    x_test_bounding_box_stretched[i] = construct_bounding_box_stretched(x_test_thresholded[i])

threshold_value = 1.8360763149212918e-10
x_train_thresholded_strech = np.where(x_train_bounding_box_stretched > threshold_value, 1, 0)
x_test_thresholded_strech = np.where(x_test_bounding_box_stretched > threshold_value, 1, 0)



# Create an SVM model
# sgd_model = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)

svm_model = SVC()
# # Testing on Test data

# # Flatten the images
x_train_thresholded_strech_flattened = x_train_thresholded_strech.reshape((x_train_thresholded_strech.shape[0], -1))
x_test_threshold_flattened = x_test_thresholded.reshape((x_test_thresholded.shape[0], -1))    

# # Train the model
svm_model.fit(x_train_thresholded_strech_flattened, y_train)

# Test the model
# y_pred = svm_model.predict(x_test_threshold_flattened)
# thres_accuracy = accuracy_score(y_test, y_pred)


# from sklearn.externals import joblib



# # Load the SVM model
# svm_model = joblib.load('svm_model.pkl')



# Create the main window
window = Tk()
window.title("Handwritten Digit Prediction")

# Create a frame to hold the sub-canvases
frame = Frame(window)
frame.pack()

# Create 6 sub-canvases
sub_canvases = []
for i in range(6):
    # Create a canvas to draw on
    canvas_width = 200
    canvas_height = 200
    canvas = Canvas(frame, width=canvas_width, height=canvas_height, bg='white')
    canvas.pack(side=LEFT)

    # Add a label to display the prediction result
    result_label = Label(frame, text="")
    result_label.pack(side=LEFT)

    sub_canvases.append((canvas, result_label))

# Create a label to show the predicted string
predicted_string_label = Label(window, text="")
predicted_string_label.pack()

def predict_digits():
    predictions = []
    for canvas, result_label in sub_canvases:
        # Get the image from the canvas
        image = canvas.postscript(colormode='color')
        image = Image.open(io.BytesIO(image.encode('utf-8')))
        image = image.resize((28, 28)).convert('L')

        # Apply thresholding
        threshold_value = 127
        image = np.array(image)
        image = np.where(image > threshold_value, 255, 0)
        image = Image.fromarray(image.astype(np.uint8))

        # Invert the colors
        image = ImageOps.invert(image)

        # Resize the image to 28x28
        image = np.array(image)
        image = resize(image, (28, 28), anti_aliasing=True)

        image = construct_bounding_box_stretched(image)

        threshold_value2 = 1.8360763149212918e-10
        image = np.where(image > threshold_value2, 1, 0)

        # Flatten the image into a 1D array
        image_array = image.flatten()

        # Make a prediction using the SVM model
        prediction = svm_model.predict([image_array])[0]
        predictions.append(prediction)

        # Update the result label
        # result_label.config(text="Prediction: " + str(prediction))

    # Update the predicted string label
    predicted_string_label.config(text="Pin Code: " + ''.join(str(p) for p in predictions))

def clear_canvases():
    for canvas, result_label in sub_canvases:
        canvas.delete("all")
        result_label.config(text="")
    predicted_string_label.config(text="")

def paint(canvas, event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill='black')

# Bind the painting event to all the canvases
for canvas, _ in sub_canvases:
    canvas.bind('<B1-Motion>', lambda event, canvas=canvas: paint(canvas, event))


# Add a button to predict the digits
predict_button = Button(window, text="Predict", command=predict_digits)
predict_button.pack()

# Add a button to clear the canvases
clear_button = Button(window, text="Clear", command=clear_canvases)
clear_button.pack()

# Start the main event loop
window.mainloop()


