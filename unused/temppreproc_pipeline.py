non_img_arr = []
can_img_arr = []

for img in NewN_IDC:

    n_img = cv2.imread(img, cv2.IMREAD_COLOR)
    n_img_size = cv2.resize(n_img, (50, 50), interpolation = cv2.INTER_LINEAR)
    non_img_arr.append([n_img_size, 0])

for img in P_IDC:
    c_img = cv2.imread(img, cv2.IMREAD_COLOR)
    c_img_size = cv2.resize(c_img, (50, 50), interpolation = cv2.INTER_LINEAR)
    can_img_arr.append([c_img_size, 1])
#CUSTOM

import numpy as np
import random

# Separate features and labels
non_features = [feature for feature, label in non_img_arr[:12389]]
non_labels = [label for feature, label in non_img_arr[:12389]]

can_features = [feature for feature, label in can_img_arr[:12389]]
can_labels = [label for feature, label in can_img_arr[:12389]]

# Concatenate features and labels
X = np.concatenate((non_features, can_features))
y = np.concatenate((non_labels, can_labels))

# Combine features and labels into a single array
breast_img_arr = list(zip(X, y))

# Shuffle the combined array
random.shuffle(breast_img_arr)

# Separate features and labels after shuffling
X, y = zip(*breast_img_arr)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

X = []
y = []

breast_img_arr = np.concatenate((non_img_arr[:12389], can_img_arr[:12389]))
random.shuffle(breast_img_arr)

for feature, label in breast_img_arr:
    X.append(feature)
    y.append(label)

X = np.array(X)
y = np.array(y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 2)
Y_test = to_categorical(Y_test, num_classes = 2)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
