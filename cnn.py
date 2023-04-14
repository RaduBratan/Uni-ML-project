# import all modules
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import os
import PIL
import tensorflow as tf
import pathlib
import pandas as pd
import gc

from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

from keras import backend as K

# define image file paths
data_img_dir = '/kaggle/input/brain-anomaly-detection2/data/data/'
data_img_dir = pathlib.Path(data_img_dir)

# define training and validation data file paths
data_training = '/kaggle/input/brain-anomaly-detection2/data/train_labels.txt'
data_validation = '/kaggle/input/brain-anomaly-detection2/data/validation_labels.txt'

# load validation data; x is for images, y is for labels
validation_data_x, validation_data_y = [], []

# loop through all the images and labels
with open(data_validation, 'r') as f:
    next(f)
    for line in f:
        img_id, label = line.strip().split(',')
        label = int(label)
        img_path = os.path.join(data_img_dir, img_id + '.png')
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        validation_data_x.append(img_gray)
        validation_data_y.append(label)

# put images and labels into their respective arrays
validation_data_x = np.array(validation_data_x, dtype='float16')
validation_data_y = np.array(validation_data_y, dtype='float16')

# load training data; x is for images, y is for labels
training_data_x, training_data_y = [], []

# loop through all the images and labels
with open(data_training, 'r') as f:
    next(f)
    for line in f:
        img_id, label = line.strip().split(',')
        label = int(label)
        img_path = os.path.join(data_img_dir, img_id + '.png')
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        training_data_x.append(img_gray)
        training_data_y.append(label)

# put images and labels into their respective arrays
training_data_x = np.array(training_data_x, dtype='float16')
training_data_y = np.array(training_data_y, dtype='float16')

# necessary garbage collection (Kaggle has limited CPU)
gc.collect()

# get data shapes
print(training_data_x.shape)
print(validation_data_x.shape)

# set data shapes
training_data_x.shape = (15000, 224, 224, 1)
validation_data_x.shape = (2000, 224, 224, 1)

# define Mean F1-Score
@tf.function
def f1(true_y, pred_y):
    precision = K.sum(K.round(K.clip(true_y * pred_y, 0, 1))) / (K.sum(K.round(K.clip(pred_y, 0, 1))) + K.epsilon())
    recall = K.sum(K.round(K.clip(true_y * pred_y, 0, 1))) / (K.sum(K.round(K.clip(true_y, 0, 1))) + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# necessary garbage collection (Kaggle has limited CPU)
gc.collect()

# define parameters for data augmentation
data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'nearest'
)

# define regularizer
regularizer = tf.keras.regularizers.L2(0.001)

# define the second CNN model
model = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, kernel_size=(3, 3), padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, kernel_size=(3, 3), padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, kernel_size=(3, 3), padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation = 'sigmoid', kernel_regularizer = regularizer),
    layers.Dropout(0.5),
    layers.Dense(32, activation = 'relu', kernel_regularizer = regularizer),
    layers.Dropout(0.5),
    layers.Dense(1, activation = 'sigmoid'),
])

# compile the model
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = [f1, 'accuracy']
)

# define when the model will stop early
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'f1',
    mode = 'max',
    patience = 10,
    verbose = 1
)

# define learning rate
learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 5,
    verbose = 1
)

# train the model
model.fit(
    data_augmentation.flow(training_data_x, training_data_y, batch_size = 32),
    epochs = 100,
    steps_per_epoch = len(training_data_x) // 32,
    validation_data = (validation_data_x, validation_data_y),
    callbacks = [learning_rate_scheduler, early_stopping]
)

# necessary garbage collection (Kaggle has limited CPU)
gc.collect()

# evaluate the model
model.evaluate(validation_data_x, validation_data_y)

# necessary garbage collection (Kaggle has limited CPU)
gc.collect()

# create testing data
data_list = list(data_img_dir.glob('*'))
data_list.sort(key = lambda x : str(x))

testing_data_x = []

for i in range(17000, 22149):
    img = cv2.imread(str(data_list[i]))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    testing_data_x.append(img_gray)

testing_data_x = np.array(testing_data_x, dtype = 'float16')
testing_data_x /= 255

# necessary garbage collection (Kaggle has limited CPU)
gc.collect()

# get testing data shape
print(testing_data_x.shape)

# set testing data shape
testing_data_x.shape = (5149, 224, 224, 1)

# necessary garbage collection (Kaggle has limited CPU)
gc.collect()

# predict the testing data
prediction_data_x = model.predict(testing_data_x)

# necessary garbage collection (Kaggle has limited CPU)
gc.collect()

# add labels (0 for normal, 1 for anomaly) based on predictions
prediction_data_x_arr = []

for element in prediction_data_x:
    if element > 0.5:
        prediction_data_x_arr.append(1)
    else:
        prediction_data_x_arr.append(0)

# write results to .csv file
with open('/kaggle/working/test_labels.csv', 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'class'])
    for i in range(17001, 22150):
        writer.writerow(['0' + str(i), prediction_data_x_arr[i - 17001]])

# necessary garbage collection (Kaggle has limited CPU)
gc.collect()