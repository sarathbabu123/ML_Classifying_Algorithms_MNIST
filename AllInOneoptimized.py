import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import math

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







# def knn_classification(x_train, y_train, x_test,train=True):
#     # Reshape the data
#     x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
#     x_test_reshaped = x_test.reshape(x_test.shape[0], -1)

#     clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
#     clf.fit(x_train_reshaped, y_train)

#     # Make predictions on the test data
#     y_pred = clf.predict(x_test_reshaped)

#     # Compute the accuracy of the predictions
#     if train:
#         accuracy = accuracy_score(y_train, y_pred)
#     else:
#         accuracy = accuracy_score(y_test, y_pred)
#     return accuracy





# test_thres_accuracy = knn_classification(x_train_thresholded, y_train, x_test_thresholded,train=False)
# test_bb_accuracy = knn_classification(x_train_bounding_box, y_train, x_test_bounding_box,train=False)
# test_bb_strech_accuracy = knn_classification(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched,train=False)
# test_thres_strech_accuracy = knn_classification(x_train_thresholded_strech, y_train, x_test_thresholded_strech,train=False)


# train_thres_accuracy = knn_classification(x_train_thresholded, y_train, x_test_thresholded)
# train_bb_accuracy = knn_classification(x_train_bounding_box, y_train, x_test_bounding_box)
# train_bb_strech_accuracy = knn_classification(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched)
# train_thres_strech_accuracy = knn_classification(x_train_thresholded_strech, y_train, x_test_thresholded_strech)


# print("KNN on TEST DATA")
# print(f"KNN 🧪 Thresholded Accuracy: {test_thres_accuracy:.4f}")
# print(f"KNN 🧪 Box Bounded Accuracy: {test_bb_accuracy:.4f}")
# print(f"KNN 🧪 Box bounding streched Accuracy: {train_bb_strech_accuracy:.4f}")
# print(f"KNN 🧪 Thresholded Box bounding streched Accuracy: {train_thres_strech_accuracy:.4f}")

# print("KNN on TRAIN DATA")
# print(f"KNN 🚂  Threshold Accuracy {train_thres_accuracy:.4f}")
# print(f"KNN 🚂  Bound Box Accuracy: {train_bb_accuracy:.4f}")
# print(f"KNN 🚂  Bound Box Streched Accuracy: {train_bb_strech_accuracy:.4f}")
# print(f"KNN 🚂  Bound Box Streched tresholded Accuracy: {train_thres_strech_accuracy:.4f}")

def knn_classification(x_train, y_train, x_test,train=True):
    # Reshape the data
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)

    clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    clf.fit(x_train_reshaped, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(x_test_reshaped)

    # Compute the accuracy of the predictions
    if train:
        accuracy = accuracy_score(y_train, y_pred)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy



test_thres_accuracy = knn_classification(x_train_thresholded, y_train, x_test_thresholded,train=False)
test_bb_accuracy = knn_classification(x_train_bounding_box, y_train, x_test_bounding_box,train=False)
test_bb_strech_accuracy = knn_classification(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched,train = False)
test_thres_strech_accuracy = knn_classification(x_train_thresholded_strech, y_train, x_test_thresholded_strech, train = False)

train_thres_accuracy = knn_classification(x_train_thresholded, y_train, x_train_thresholded)
train_bb_accuracy = knn_classification(x_train_bounding_box, y_train, x_train_bounding_box)
train_bb_strech_accuracy = knn_classification(x_train_bounding_box_stretched, y_train, x_train_bounding_box_stretched)
train_thres_strech_accuracy = knn_classification(x_train_thresholded_strech, y_train, x_train_thresholded_strech)

print("TESTING ON TEST")
print(f"Thresholded KNN Accuracy: {test_thres_accuracy:.4f}")
print(f"Thresholded Box Bounded KNN Accuracy: {test_bb_accuracy:.4f}")
print(f"Thresholded Box bounding streched KNN Accuracy: {test_bb_strech_accuracy:.4f}")
print(f"Thresholded Box bounding streched KNN Accuracy: {test_thres_strech_accuracy:.4f}")

print("TESTING ON TRAIN")
print(f"Thresholded KNN Accuracy: {train_thres_accuracy:.4f}")
print(f"Thresholded Box Bounded KNN Accuracy: {train_bb_accuracy:.4f}")
print(f"Thresholded Box bounding streched KNN Accuracy: {train_bb_strech_accuracy:.4f}")
print(f"Thresholded Box bounding streched KNN Accuracy: {train_thres_strech_accuracy:.4f}")


def naive_bayes_classification_Gaussian(x_train, y_train, x_test, train=True):
    # Reshape the data
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_data_reshaped = x_test.reshape(x_test.shape[0], -1)

    clf = GaussianNB()
    clf.fit(x_train_reshaped, y_train)

    # Make predictions on the data
    y_pred = clf.predict(x_data_reshaped)

    # Compute the accuracy of the predictions
    if train:
        accuracy = accuracy_score(y_train, y_pred)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    return accuracy

test_naive_thres_accuracy = naive_bayes_classification_Gaussian(x_train_thresholded, y_train, x_test_thresholded,train=False)
test_naive_bb_accuracy = naive_bayes_classification_Gaussian(x_train_bounding_box, y_train, x_test_bounding_box,train=False)
test_naive_bb_strech_accuracy = naive_bayes_classification_Gaussian(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched,train=False)
test_naive_thres_strech_accuracy = naive_bayes_classification_Gaussian(x_train_thresholded_strech, y_train, x_test_thresholded_strech,train=False)

train_naive_thres_accuracy = naive_bayes_classification_Gaussian(x_train_thresholded, y_train, x_train_thresholded)
train_naive_bb_accuracy = naive_bayes_classification_Gaussian(x_train_bounding_box, y_train, x_train_bounding_box)
train_naive_bb_strech_accuracy = naive_bayes_classification_Gaussian(x_train_bounding_box_stretched, y_train, x_train_bounding_box_stretched)
train_naive_thres_strech_accuracy = naive_bayes_classification_Gaussian(x_train_thresholded_strech, y_train, x_train_thresholded_strech)

print("GNB on TEST DATA")
print(f"GNB 🧪 Threshold Accuracy {test_naive_thres_accuracy:.4f}")
print(f"GNB 🧪 Bound Box Accuracy: {test_naive_bb_accuracy:.4f}")
print(f"GNB 🧪 Bound Box Streched Accuracy: {test_naive_bb_strech_accuracy:.4f}")
print(f"GNB 🧪 Bound Box Streched tresholded Accuracy: {test_naive_thres_strech_accuracy:.4f}")

print("GNB on TRAIN DATA")
print(f"GNB 🚂  Threshold Accuracy {train_naive_thres_accuracy:.4f}")
print(f"GNB 🚂  Bound Box Accuracy: {train_naive_bb_accuracy:.4f}")
print(f"GNB 🚂  Bound Box Streched Accuracy: {train_naive_bb_strech_accuracy:.4f}")
print(f"GNB 🚂  Bound Box Streched tresholded Accuracy: {train_naive_thres_strech_accuracy:.4f}")


def naive_bayes_classification_bernoulli(x_train, y_train, x_test, train=True):
    # Reshape the data
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_data_reshaped = x_test.reshape(x_test.shape[0], -1)

    clf = BernoulliNB()
    clf.fit(x_train_reshaped, y_train)

    # Make predictions on the data
    y_pred = clf.predict(x_data_reshaped)

    # Compute the accuracy of the predictions
    if train:
        accuracy = accuracy_score(y_train, y_pred)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    return accuracy


test_naive_thres_accuracy = naive_bayes_classification_bernoulli(x_train_thresholded, y_train, x_test_thresholded,train=False)
test_naive_bb_accuracy = naive_bayes_classification_bernoulli(x_train_bounding_box, y_train, x_test_bounding_box,train=False)
test_naive_bb_strech_accuracy = naive_bayes_classification_bernoulli(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched,train=False)
test_naive_thres_strech_accuracy = naive_bayes_classification_bernoulli(x_train_thresholded_strech, y_train, x_test_thresholded_strech,train=False)

train_naive_thres_accuracy = naive_bayes_classification_bernoulli(x_train_thresholded, y_train, x_train_thresholded)
train_naive_bb_accuracy = naive_bayes_classification_bernoulli(x_train_bounding_box, y_train, x_train_bounding_box)
train_naive_bb_strech_accuracy = naive_bayes_classification_bernoulli(x_train_bounding_box_stretched, y_train, x_train_bounding_box_stretched)
train_naive_thres_strech_accuracy = naive_bayes_classification_bernoulli(x_train_thresholded_strech, y_train, x_train_thresholded_strech)

print("BNB on TEST DATA")
print(f"BNB 🧪 Thresholded Accuracy: {test_naive_thres_accuracy:.4f}")
print(f"BNB 🧪 Bound Box Accuracy: {test_naive_bb_accuracy:.4f}")
print(f"BNB 🧪 Bound Box Streched Accuracy: {test_naive_bb_strech_accuracy:.4f}")
print(f"BNB 🧪 Bound Box Streched tresholded Accuracy: {test_naive_thres_strech_accuracy:.4f}")

print("BNB on TRAIN DATA")
print(f"BNB 🚂  Threshold Accuracy {train_naive_thres_accuracy:.4f}")
print(f"BNB 🚂  Bound Box Accuracy: {train_naive_bb_accuracy:.4f}")
print(f"BNB 🚂  Bound Box Streched Accuracy: {train_naive_bb_strech_accuracy:.4f}")
print(f"BNB 🚂  Bound Box Streched tresholded Accuracy: {train_naive_thres_strech_accuracy:.4f}")


# def SVM_classifier(x_train, y_train, x_test, train=True):
#     # Reshape the data
#     x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
#     x_data_reshaped = x_test.reshape(x_test.shape[0], -1)

#     clf = SVC()
#     clf.fit(x_train_reshaped, y_train)

#     # Make predictions on the data
#     y_pred = clf.predict(x_data_reshaped)

#     # Compute the accuracy of the predictions
#     if train:
#         accuracy = accuracy_score(y_train, y_pred)
#     else:
#         accuracy = accuracy_score(y_test, y_pred)
#     return accuracy


# test_SVM_thres_accuracy = SVM_classifier(x_train_thresholded, y_train, x_test_thresholded,train=False)
# test_SVM_bb_accuracy = SVM_classifier(x_train_bounding_box, y_train, x_test_bounding_box,train=False)
# test_SVM_bb_strech_accuracy = SVM_classifier(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched,train=False)
# test_SVM_thres_strech_accuracy = SVM_classifier(x_train_thresholded_strech, y_train, x_test_thresholded_strech,train=False)

# train_SVM_thres_accuracy = SVM_classifier(x_train_thresholded, y_train, x_train_thresholded)
# train_SVM_bb_accuracy = SVM_classifier(x_train_bounding_box, y_train, x_train_bounding_box)
# train_SVM_bb_strech_accuracy = SVM_classifier(x_train_bounding_box_stretched, y_train, x_train_bounding_box_stretched)
# train_SVM_thres_strech_accuracy = SVM_classifier(x_train_thresholded_strech, y_train, x_train_thresholded_strech)

# print("SVM on TEST DATA")

# print(f"SVM 🧪 Thresholded Accuracy: {test_SVM_thres_accuracy:.4f}")
# print(f"SVM 🧪 Bound Box Accuracy: {test_SVM_bb_accuracy:.4f}")
# print(f"SVM 🧪 Bound Box Streched Accuracy: {test_SVM_bb_strech_accuracy:.4f}")
# print(f"SVM 🧪 Bound Box Streched tresholded Accuracy: {test_SVM_thres_strech_accuracy:.4f}")

# print("SVM on TRAIN DATA")
# print(f"SVM 🚂  Threshold Accuracy {train_SVM_thres_accuracy:.4f}")
# print(f"SVM 🚂  Bound Box Accuracy: {train_SVM_bb_accuracy:.4f}")
# print(f"SVM 🚂  Bound Box Streched Accuracy: {train_SVM_bb_strech_accuracy:.4f}")
# print(f"SVM 🚂  Bound Box Streched tresholded Accuracy: {train_SVM_thres_strech_accuracy:.4f}")

def RF_10T_4D_classifier(x_train, y_train, x_test, train=True):
    # Reshape the data
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_data_reshaped = x_test.reshape(x_test.shape[0], -1)

    rfc_model = RandomForestClassifier(n_estimators=10, max_depth=4)
    rfc_model.fit(x_train_reshaped, y_train)

    # Make predictions on the data
    y_pred = rfc_model.predict(x_data_reshaped)

    # Compute the accuracy of the predictions
    if train:
        accuracy = accuracy_score(y_train, y_pred)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    return accuracy


test_RF_10T_4D_thres_accuracy = RF_10T_4D_classifier(x_train_thresholded, y_train, x_test_thresholded,train=False)
test_RF_10T_4D_bb_accuracy = RF_10T_4D_classifier(x_train_bounding_box, y_train, x_test_bounding_box,train=False)
test_RF_10T_4D_bb_strech_accuracy = RF_10T_4D_classifier(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched,train=False)
test_RF_10T_4D_thres_strech_accuracy = RF_10T_4D_classifier(x_train_thresholded_strech, y_train, x_test_thresholded_strech,train=False)

train_RF_10T_4D_thres_accuracy = RF_10T_4D_classifier(x_train_thresholded, y_train, x_train_thresholded)
train_RF_10T_4D_bb_accuracy = RF_10T_4D_classifier(x_train_bounding_box, y_train, x_train_bounding_box)
train_RF_10T_4D_bb_strech_accuracy = RF_10T_4D_classifier(x_train_bounding_box_stretched, y_train, x_train_bounding_box_stretched)
train_RF_10T_4D_thres_strech_accuracy = RF_10T_4D_classifier(x_train_thresholded_strech, y_train, x_train_thresholded_strech)

print("RF 10T 4D on TEST DATA")
print(f"RF 10T 4D 🧪 Thresholded Accuracy: {test_RF_10T_4D_thres_accuracy:.4f}")
print(f"RF 10T 4D 🧪 Bound Box Accuracy: {test_RF_10T_4D_bb_accuracy:.4f}")
print(f"RF 10T 4D 🧪 Bound Box Streched Accuracy: {test_RF_10T_4D_bb_strech_accuracy:.4f}")
print(f"RF 10T 4D 🧪 Bound Box Streched tresholded Accuracy: {test_RF_10T_4D_thres_strech_accuracy:.4f}")

print("RF 10T 4D on TRAIN DATA")
print(f"RF 10T 4D 🚂  Threshold Accuracy {train_RF_10T_4D_thres_accuracy:.4f}")
print(f"RF 10T 4D 🚂  Bound Box Accuracy: {train_RF_10T_4D_bb_accuracy:.4f}")
print(f"RF 10T 4D 🚂  Bound Box Streched Accuracy: {train_RF_10T_4D_bb_strech_accuracy:.4f}")
print(f"RF 10T 4D 🚂  Bound Box Streched tresholded Accuracy: {train_RF_10T_4D_thres_strech_accuracy:.4f}")


def RF_10T_16D_classifier(x_train, y_train, x_test, train=True):
    # Reshape the data
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_data_reshaped = x_test.reshape(x_test.shape[0], -1)

    rfc_model = RandomForestClassifier(n_estimators=10, max_depth=16)
    rfc_model.fit(x_train_reshaped, y_train)

    # Make predictions on the data
    y_pred = rfc_model.predict(x_data_reshaped)

    # Compute the accuracy of the predictions
    if train:
        accuracy = accuracy_score(y_train, y_pred)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    return accuracy


test_RF_10T_16D_thres_accuracy = RF_10T_16D_classifier(x_train_thresholded, y_train, x_test_thresholded,train=False)
test_RF_10T_16D_bb_accuracy = RF_10T_16D_classifier(x_train_bounding_box, y_train, x_test_bounding_box,train=False)
test_RF_10T_16D_bb_strech_accuracy = RF_10T_16D_classifier(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched,train=False)
test_RF_10T_16D_thres_strech_accuracy = RF_10T_16D_classifier(x_train_thresholded_strech, y_train, x_test_thresholded_strech,train=False)

train_RF_10T_16D_thres_accuracy = RF_10T_16D_classifier(x_train_thresholded, y_train, x_train_thresholded)
train_RF_10T_16D_bb_accuracy = RF_10T_16D_classifier(x_train_bounding_box, y_train, x_train_bounding_box)
train_RF_10T_16D_bb_strech_accuracy = RF_10T_16D_classifier(x_train_bounding_box_stretched, y_train, x_train_bounding_box_stretched)
train_RF_10T_16D_thres_strech_accuracy = RF_10T_16D_classifier(x_train_thresholded_strech, y_train, x_train_thresholded_strech)


print("RF 10T 16D on TEST DATA")
print(f"RF 10T 16D 🧪 Thresholded Accuracy: {test_RF_10T_16D_thres_accuracy:.4f}")
print(f"RF 10T 16D 🧪 Bound Box Accuracy: {test_RF_10T_16D_bb_accuracy:.4f}")
print(f"RF 10T 16D 🧪 Bound Box Streched Accuracy: {test_RF_10T_16D_bb_strech_accuracy:.4f}")
print(f"RF 10T 16D 🧪 Bound Box Streched tresholded Accuracy: {test_RF_10T_16D_thres_strech_accuracy:.4f}")

print("RF 10T 16D on TRAIN DATA")
print(f"RF 10T 16D 🚂  Threshold Accuracy {train_RF_10T_16D_thres_accuracy:.4f}")
print(f"RF 10T 16D 🚂  Bound Box Accuracy: {train_RF_10T_16D_bb_accuracy:.4f}")
print(f"RF 10T 16D 🚂  Bound Box Streched Accuracy: {train_RF_10T_16D_bb_strech_accuracy:.4f}")
print(f"RF 10T 16D 🚂  Bound Box Streched tresholded Accuracy: {train_RF_10T_16D_thres_strech_accuracy:.4f}")


def RF_30T_4D_classifier(x_train, y_train, x_test, train=True):
    # Reshape the data
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_data_reshaped = x_test.reshape(x_test.shape[0], -1)

    rfc_model = RandomForestClassifier(n_estimators=30, max_depth=4)
    rfc_model.fit(x_train_reshaped, y_train)

    # Make predictions on the data
    y_pred = rfc_model.predict(x_data_reshaped)

    # Compute the accuracy of the predictions
    if train:
        accuracy = accuracy_score(y_train, y_pred)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    return accuracy

test_RF_30T_4D_thres_accuracy = RF_30T_4D_classifier(x_train_thresholded, y_train, x_test_thresholded,train=False)
test_RF_30T_4D_bb_accuracy = RF_30T_4D_classifier(x_train_bounding_box, y_train, x_test_bounding_box,train=False)
test_RF_30T_4D_bb_strech_accuracy = RF_30T_4D_classifier(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched,train=False)
test_RF_30T_4D_thres_strech_accuracy = RF_30T_4D_classifier(x_train_thresholded_strech, y_train, x_test_thresholded_strech,train=False)

train_RF_30T_4D_thres_accuracy = RF_30T_4D_classifier(x_train_thresholded, y_train, x_train_thresholded)
train_RF_30T_4D_bb_accuracy = RF_30T_4D_classifier(x_train_bounding_box, y_train, x_train_bounding_box)
train_RF_30T_4D_bb_strech_accuracy = RF_30T_4D_classifier(x_train_bounding_box_stretched, y_train, x_train_bounding_box_stretched)
train_RF_30T_4D_thres_strech_accuracy = RF_30T_4D_classifier(x_train_thresholded_strech, y_train, x_train_thresholded_strech)

print("RF 30T 4D on TEST DATA")
print(f"RF 30T 4D 🧪 Thresholded Accuracy: {test_RF_30T_4D_thres_accuracy:.4f}")
print(f"RF 30T 4D 🧪 Bound Box Accuracy: {test_RF_30T_4D_bb_accuracy:.4f}")
print(f"RF 30T 4D 🧪 Bound Box Streched Accuracy: {test_RF_30T_4D_bb_strech_accuracy:.4f}")
print(f"RF 30T 4D 🧪 Bound Box Streched tresholded Accuracy: {test_RF_30T_4D_thres_strech_accuracy:.4f}")

print("RF 30T 4D on TRAIN DATA")
print(f"RF 30T 4D 🚂  Threshold Accuracy {train_RF_30T_4D_thres_accuracy:.4f}")
print(f"RF 30T 4D 🚂  Bound Box Accuracy: {train_RF_30T_4D_bb_accuracy:.4f}")
print(f"RF 30T 4D 🚂  Bound Box Streched Accuracy: {train_RF_30T_4D_bb_strech_accuracy:.4f}")
print(f"RF 30T 4D 🚂  Bound Box Streched tresholded Accuracy: {train_RF_30T_4D_thres_strech_accuracy:.4f}")


def RF_30T_16D_classifier(x_train, y_train, x_test, train=True):
    # Reshape the data
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_data_reshaped = x_test.reshape(x_test.shape[0], -1)

    rfc_model = RandomForestClassifier(n_estimators=30, max_depth=16)
    rfc_model.fit(x_train_reshaped, y_train)

    # Make predictions on the data
    y_pred = rfc_model.predict(x_data_reshaped)

    # Compute the accuracy of the predictions
    if train:
        accuracy = accuracy_score(y_train, y_pred)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    return accuracy


test_RF_30T_16D_thres_accuracy = RF_30T_16D_classifier(x_train_thresholded, y_train, x_test_thresholded,train=False)
test_RF_30T_16D_bb_accuracy = RF_30T_16D_classifier(x_train_bounding_box, y_train, x_test_bounding_box,train=False)
test_RF_30T_16D_bb_strech_accuracy = RF_30T_16D_classifier(x_train_bounding_box_stretched, y_train, x_test_bounding_box_stretched,train=False)
test_RF_30T_16D_thres_strech_accuracy = RF_30T_16D_classifier(x_train_thresholded_strech, y_train, x_test_thresholded_strech,train=False)

train_RF_30T_16D_thres_accuracy = RF_30T_16D_classifier(x_train_thresholded, y_train, x_train_thresholded)
train_RF_30T_16D_bb_accuracy = RF_30T_16D_classifier(x_train_bounding_box, y_train, x_train_bounding_box)
train_RF_30T_16D_bb_strech_accuracy = RF_30T_16D_classifier(x_train_bounding_box_stretched, y_train, x_train_bounding_box_stretched)
train_RF_30T_16D_thres_strech_accuracy = RF_30T_16D_classifier(x_train_thresholded_strech, y_train, x_train_thresholded_strech)


print("RF 30T 16D on TEST DATA")
print(f"RF 30T 16D 🧪 Thresholded Accuracy: {test_RF_30T_16D_thres_accuracy:.4f}")
print(f"RF 30T 16D 🧪 Bound Box Accuracy: {test_RF_30T_16D_bb_accuracy:.4f}")
print(f"RF 30T 16D 🧪 Bound Box Streched Accuracy: {test_RF_30T_16D_bb_strech_accuracy:.4f}")
print(f"RF 30T 16D 🧪 Bound Box Streched tresholded Accuracy: {test_RF_30T_16D_thres_strech_accuracy:.4f}")

print("RF 30T 16D on TRAIN DATA")
print(f"RF 30T 16D 🚂  Threshold Accuracy {train_RF_30T_16D_thres_accuracy:.4f}")
print(f"RF 30T 16D 🚂  Bound Box Accuracy: {train_RF_30T_16D_bb_accuracy:.4f}")
print(f"RF 30T 16D 🚂  Bound Box Streched Accuracy: {train_RF_30T_16D_bb_strech_accuracy:.4f}")
print(f"RF 30T 16D 🚂  Bound Box Streched tresholded Accuracy: {train_RF_30T_16D_thres_strech_accuracy:.4f}")