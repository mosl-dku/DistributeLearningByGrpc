import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np

class conmodel:
    def __init__(self):
        self.result = 0

    def model():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',padding="same", input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu',padding="same" ))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3),padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3),padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(512, (3, 3),padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        return model

    def model1(input_data):

    
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',padding="same", input_shape=input_data[1].shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu',padding="same" ))
        model.add(layers.MaxPooling2D((2, 2)))

        return model
    
    def model1_1(input_data):

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',padding="same", input_shape=input_data[1].shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))

        return model

    def model2():

        model = models.Sequential()
        model.add(layers.Conv2D(128, (3, 3),padding="same", activation='relu', input_shape=(8, 8, 64)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3),padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(512, (3, 3),padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        return model

    def model2_2():

        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu',padding="same", input_shape=(16, 16, 32) ))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3),padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3),padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(512, (3, 3),padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        return model

    