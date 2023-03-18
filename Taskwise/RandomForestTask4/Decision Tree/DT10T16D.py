#######################DECISION TREE 16 Depth ##############################################################

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier




# Load the MNIST dataset
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




tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=16)



# Flatten the images
x_train_threshold_flattened = x_train_thresholded.reshape((x_train_thresholded.shape[0], -1))
x_test_threshold_flattened = x_test_thresholded.reshape((x_test_thresholded.shape[0], -1))    

# Train the model
tree_clf.fit(x_train_threshold_flattened, y_train)

# Test the model
y_pred = tree_clf.predict(x_test_threshold_flattened)
thres_accuracy = accuracy_score(y_test, y_pred)

print("RF 10T,4D on TEST DATASET")

print(f"RF 10T,4D 🧪 Thresholded Accuracy: {thres_accuracy:.4f}")

### RF 10T,4D on box bounding dataset.

x_train_bound_box_flattened = x_train_bounding_box.reshape((x_train_bounding_box.shape[0], -1))
x_test_bound_box_flattened = x_test_bounding_box.reshape((x_test_bounding_box.shape[0], -1))

tree_clf.fit(x_train_bound_box_flattened, y_train)

y_pred = tree_clf.predict(x_test_bound_box_flattened)
bb_accuracy = accuracy_score(y_test, y_pred)

print(f"RF 10T,4D 🧪 Bound Box Accuracy: {bb_accuracy:.4f}")

## RF 10T,4D on box bounding streched dataset.

x_train_bound_box_streched_flattened = x_train_bounding_box_stretched.reshape((x_train_bounding_box_stretched.shape[0], -1))
x_test_bound_box_streched_flattened = x_test_bounding_box_stretched.reshape((x_test_bounding_box_stretched.shape[0], -1))

tree_clf.fit(x_train_bound_box_streched_flattened, y_train)

y_pred = tree_clf.predict(x_test_bound_box_streched_flattened)
bbs_accuracy = accuracy_score(y_test, y_pred)

print(f"RF 10T,4D 🧪 Bound Box Streched Accuracy: {bbs_accuracy:.4f}")

### RF 10T,4D on box bounding streched threshold dataset.

x_train_bound_box_streched_threshold_flattened = x_train_thresholded_strech.reshape((x_train_thresholded_strech.shape[0], -1))
x_test_bound_box_streched_threshold_flattened = x_test_thresholded_strech.reshape((x_test_thresholded_strech.shape[0], -1))

tree_clf.fit(x_train_bound_box_streched_threshold_flattened, y_train)

y_pred = tree_clf.predict(x_test_bound_box_streched_threshold_flattened)

bbs_thres_accuracy = accuracy_score(y_test, y_pred)

print(f"RF 10T,4D 🧪 Bound Box Streched tresholded Accuracy: {bbs_thres_accuracy:.4f}")
# Testing on Train data
### RF 10T,4D on threshold dataset.

tree_clf.fit(x_train_threshold_flattened, y_train)

y_pred = tree_clf.predict(x_train_threshold_flattened)
thres_accuracy = accuracy_score(y_train, y_pred)

print("RF 10T,4D on TRAIN DATASET")

print(f"RF 10T,4D 🚂 Thresholded Accuracy: {thres_accuracy:.4f}")

### RF 10T,4D on boundbox dataset.
tree_clf.fit(x_train_bound_box_flattened, y_train)

y_pred = tree_clf.predict(x_train_bound_box_flattened)
bb_accuracy = accuracy_score(y_train, y_pred)

print(f"RF 10T,4D 🚂 Bound Box Accuracy: {bb_accuracy:.4f}")

## RF 10T,4D on boundbox streched dataset.

tree_clf.fit(x_train_bound_box_streched_flattened, y_train)

y_pred = tree_clf.predict(x_train_bound_box_streched_flattened)
bbs_accuracy = accuracy_score(y_train, y_pred)

print(f"RF 10T,4D 🚂 Bound Box Streched Accuracy: {bbs_accuracy:.4f}")

### RF 10T,4D on boundbox streched threshold dataset.
tree_clf.fit(x_train_bound_box_streched_threshold_flattened, y_train)

y_pred = tree_clf.predict(x_train_bound_box_streched_threshold_flattened)
bbs_thres_accuracy = accuracy_score(y_train, y_pred)

print(f"RF 10T,4D 🚂 Bound Box Streched threshold Accuracy: {bbs_thres_accuracy:.4f}")

