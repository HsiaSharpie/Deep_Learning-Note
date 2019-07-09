import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Flatten

class LeNet():
    @staticmethod
    def build_model(input_shape, classes):
        # Initialize the model
        model = Sequential()

        # first layer, Convolution -> Relu -> Pooling
        model.add(Conv2D(input_shape=input_shape, kernel_size=(5, 5), filters=6, strides=(1,1)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second layer, Convolution -> Relu -> Pooling
        model.add(Conv2D(kernel_size=(5, 5), filters=16, strides=(1, 1)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Fully connection layer
        model.add(Flatten())
        model.add(Dense(120, activation = 'tanh'))
        model.add(Dense(84, activation = 'tanh'))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
