import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.datasets import mnist


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

# BOUNDING BOX


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


# for i in range(len(x_train_thresholded)):
#     i = random.randint(1, 60000)
#     train_bounding_box = construct_bounding_box(x_train_thresholded[i])
#     train_bounding_box_stretched = construct_bounding_box_stretched(x_train_thresholded[i])

#     # Display the original image, thresholded image, and stretched bounding box image
#     fig, axs = plt.subplots(1, 4, figsize=(7, 3))
#     axs[0].imshow(x_train[i], cmap='gray')
#     axs[0].set_title('Original Image')
#     axs[1].imshow(x_train_thresholded[i], cmap='gray')
#     axs[1].set_title('Thresholded Image')
#     axs[2].imshow(train_bounding_box, cmap='gray')
#     axs[2].set_title('Bounding Box Image')
#     axs[3].imshow(train_bounding_box_stretched, cmap='gray')
#     axs[3].set_title('Stretched Bounding Box Image')
#     plt.show()
# for i in range(len(x_test_thresholded)):
#     test_bounding_box = construct_bounding_box(x_test_thresholded[i])
#     test_bounding_box_stretched = construct_bounding_box_stretched(
#         x_test_thresholded[i])

#     # Display the original image, thresholded image, and stretched bounding box image
#     fig, axs = plt.subplots(1, 4, figsize=(7, 3))
#     axs[0].imshow(x_test[i], cmap='gray')
#     axs[0].set_title('Original Image')
#     axs[1].imshow(x_test_thresholded[i], cmap='gray')
#     axs[1].set_title('Thresholded Image')
#     axs[2].imshow(test_bounding_box, cmap='gray')
#     axs[2].set_title('Bounding Box Image')
#     axs[3].imshow(test_bounding_box_stretched, cmap='gray')
#     axs[3].set_title('Stretched Bounding Box Image')
#     plt.show()

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




# # # Plot some example images
# fig, axs = plt.subplots(4, 4, figsize=(10, 5))
# axs = axs.ravel()
# for i in range(4):
#     axs[i].imshow(x_train[i], cmap='gray')
#     axs[i].set_title('Original Image')
#     axs[i+5].imshow(x_train_thresholded[i], cmap='gray')
#     axs[i+5].set_title('Thresholded Image')
#     axs[i+10].imshow(x_train_bounding_box[i], cmap='gray')
#     axs[i+10].set_title('Bounding Box Image')
#     axs[i+15].imshow(x_train_bounding_box_stretched[i], cmap='gray')
#     axs[i+15].set_title('Stretch Bound Box')
# plt.subplots_adjust(wspace=1, hspace=1)
# plt.show()

threshold_value = 1.8360763149212918e-10
x_train_thresholded_strech = np.where(x_train_bounding_box_stretched > threshold_value, 1, 0)
x_test_thresholded_strech = np.where(x_test_bounding_box_stretched > threshold_value, 1, 0)


print(x_train_thresholded_strech.shape)

number_to_print = 1

# get the indices of the images with the chosen number
indices = [i for i, y in enumerate(y_train) if y == number_to_print][:5]

fig, axs = plt.subplots(2, 5, figsize=(10, 5))
axs = axs.ravel()
for i, idx in enumerate(indices):
    
    axs[i].imshow(x_train_bounding_box_stretched[idx], cmap='gray')
    axs[i].set_title('Bounding Box Strech Image')
    axs[i+5].imshow(x_train_thresholded_strech[idx], cmap='gray')
    axs[i+5].set_title('Treshold Stretch Bound Box')
plt.subplots_adjust(wspace=0.3, hspace=1)
plt.show()
