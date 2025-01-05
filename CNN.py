from helper_functions import *
import logging, os
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split

logging.getLogger('tensorflow').disabled = True
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
from tensorflow.keras.regularizers import l2

IMG_SIZE = (240, 240)

def get_CNN_model(optimizer, l2_reg = 0.01):
  model = Sequential([
    Input((IMG_SIZE[0],IMG_SIZE[1] ,3)),

    # First convolutional block
    Conv2D(128, 7, activation='relu', kernel_regularizer=l2(l2_reg)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Second convolutional block
    Conv2D(64, 7, activation='relu', kernel_regularizer=l2(l2_reg)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Third convolutional block
    Conv2D(64, 7, activation='relu', kernel_regularizer=l2(l2_reg)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Forth convolutional block
    Conv2D(32, 7, activation='relu', kernel_regularizer=l2(l2_reg)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Flatten
    Flatten(),
    Dense(1, activation='sigmoid')
  ])
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

class CNN_optimization:
    def __init__(self):
        # load data
        X, y = load_data()
        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(X, y, test_size=0.2)

    def get_metrics(self, N: np.ndarray) -> tuple:
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=N[0],
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=N[3])
        model = get_CNN_model(optimizer, l2_reg = N[2])
        return train(model, self.X_train, self.X_test, self.y_train, self.y_test, epochs=int(N[1]), verbose=False)
