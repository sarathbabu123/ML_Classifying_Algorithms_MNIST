################################################KNearestNeighbors#####################################################
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.pyplot as plt
import timeit
from keras.datasets import mnist
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import math

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define the threshold value
threshold_value = 127

# Threshold the images
x_train_thresholded = np.where(x_train > threshold_value, 1, 0)
x_test_thresholded = np.where(x_test > threshold_value, 1, 0)

# # Plot some example images
# fig, axs = plt.subplots(2, 5, figsize=(10, 5))
# axs = axs.ravel()
# for i in range(5):
#     axs[i].imshow(x_train[i], cmap='gray')
#     axs[i].set_title('Original Image')
#     axs[i+5].imshow(x_train_thresholded[i], cmap='gray')
#     axs[i+5].set_title('Thresholded Image')
# plt.show()

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



# print(len(x_train_bounding_box))
# print(math.sqrt(len(x_train_bounding_box)))
print(y_train[0])

# threshold the streched data

threshold_value = 1.8360763149212918e-10
x_train_thresholded_strech = np.where(x_train_bounding_box_stretched > threshold_value, 1, 0)
x_test_thresholded_strech = np.where(x_test_bounding_box_stretched > threshold_value, 1, 0)
# print(x_test_thresholded_strech)



clf = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
# KNN for threshold
x_train_thresholded_reshaped = x_train_thresholded.reshape(x_train_thresholded.shape[0], -1)
x_test_thresholded_reshaped = x_test_thresholded.reshape(x_test_thresholded.shape[0], -1)
start = timeit.default_timer()
clf.fit(x_train_thresholded_reshaped, y_train)
y_pred = clf.predict(x_test_thresholded_reshaped)
thres_accuracy = np.sum(y_pred == y_test) / len(y_test)
stop = timeit.default_timer()
thres_time = stop - start

# KNN for bounding box
x_train_bounding_box_reshaped = x_train_bounding_box.reshape(x_train_bounding_box.shape[0], -1)
x_test_bounding_box_reshaped = x_test_bounding_box.reshape(x_test_bounding_box.shape[0], -1)
start = timeit.default_timer()
clf.fit(x_train_bounding_box_reshaped, y_train)
y_pred = clf.predict(x_test_bounding_box_reshaped)
bb_accuracy = np.sum(y_pred == y_test) / len(y_test)
stop = timeit.default_timer()
bb_time = stop - start

# KNN for bounding box stretched
x_train_bounding_box_stretched_reshaped = x_train_bounding_box_stretched.reshape(x_train_bounding_box_stretched.shape[0], -1)
x_test_bounding_box_stretched_reshaped = x_test_bounding_box_stretched.reshape(x_test_bounding_box_stretched.shape[0], -1)
start = timeit.default_timer()
clf.fit(x_train_bounding_box_stretched_reshaped, y_train)
y_pred = clf.predict(x_test_bounding_box_stretched_reshaped)
bb_strech_accuracy = np.sum(y_pred == y_test) / len(y_test)
stop = timeit.default_timer()
bb_strech_time = stop - start

# KNN for thresholded stretched
x_train_thresholded_strech_reshaped = x_train_thresholded_strech.reshape(x_train_thresholded_strech.shape[0], -1)
x_test_thresholded_strech_reshaped = x_test_thresholded_strech.reshape(x_test_thresholded_strech.shape[0], -1)
start = timeit.default_timer()
clf.fit(x_train_thresholded_strech_reshaped, y_train)
y_pred = clf.predict(x_test_thresholded_strech_reshaped)
thres_strech_accuracy = np.sum(y_pred == y_test) / len(y_test)
stop = timeit.default_timer()
thres_strech_time = stop - start

# Print the results
print(f"🧪 Thresholded KNN Accuracy: {thres_accuracy:.4f} Time: {thres_time:.4f} seconds")
print(f"🧪 Bounding Box KNN Accuracy: {bb_accuracy:.4f} Time: {bb_time:.4f} seconds")
print(f"🧪 Bounding Box Stretched KNN Accuracy: {bb_strech_accuracy:.4f} Time: {bb_strech_time:.4f} seconds")
print(f"🧪 Thresholded Stretched KNN Accuracy: {thres_strech_accuracy:.4f} Time: {thres_strech_time:.4f} seconds")



########################################################################## FOR TRAINING DATA ##########################################################

# Taining and testing on training data itself

# Reshape the thresholded data
x_train_thresholded_reshaped = x_train_thresholded.reshape(x_train_thresholded.shape[0], -1)
x_test_thresholded_reshaped = x_test_thresholded.reshape(x_test_thresholded.shape[0], -1)

clf.fit(x_train_thresholded_reshaped, y_train)

# Make predictions on the test data
y_pred = clf.predict(x_train_thresholded_reshaped)

# Compute the accuracy of the predictions
thres_accuracy = np.sum(y_pred == y_train) / len(y_train)

print(f"🚂 Thresholded KNN Accuracy: {thres_accuracy:.4f}")


### KNN for bounding box

# Reshape the bounding box data
x_train_bounding_box_reshaped = x_train_bounding_box.reshape(x_train_bounding_box.shape[0], -1)
x_test_bounding_box_reshaped = x_test_bounding_box.reshape(x_test_bounding_box.shape[0], -1)

clf.fit(x_train_bounding_box_reshaped, y_train)

# Make predictions on the test data
y_pred = clf.predict(x_train_bounding_box_reshaped)

# Compute the accuracy of the predictions
bb_accuracy = np.sum(y_pred == y_train) / len(y_train)

# print
print(f" Thresholded Box Bounded KNN Accuracy: {bb_accuracy:.4f}")

### KNN for bounding box stretched

# Reshape the bounding box data

x_train_bounding_box_stretched_reshaped = x_train_bounding_box_stretched.reshape(x_train_bounding_box_stretched.shape[0], -1)
x_test_bounding_box_stretched_reshaped = x_test_bounding_box_stretched.reshape(x_test_bounding_box_stretched.shape[0], -1)



clf.fit(x_train_bounding_box_stretched_reshaped, y_train)

# Make predictions on the test data
y_pred = clf.predict(x_train_bounding_box_stretched_reshaped)

# Compute the accuracy of the predictions

bb_strech_accuracy = np.sum(y_pred == y_train) / len(y_train)

print(f"🚂 Non Thresholded Box bounding streched KNN Accuracy: {bb_strech_accuracy:.4f}")

# Reshape the thresholded data

x_train_thresholded_strech_reshaped = x_train_thresholded_strech.reshape(x_train_thresholded_strech.shape[0], -1)
x_test_thresholded_strech_reshaped = x_test_thresholded_strech.reshape(x_test_thresholded_strech.shape[0], -1)

clf.fit(x_train_thresholded_strech_reshaped, y_train)

# Make predictions on the test data
y_pred = clf.predict(x_train_thresholded_strech_reshaped)

# Compute the accuracy of the predictions
thres_strech_accuracy = np.sum(y_pred == y_train) / len(y_train)
#print accuracy


print(f"🚂 Thresholded Box bounding streched KNN Accuracy: {thres_strech_accuracy:.4f}")



    






