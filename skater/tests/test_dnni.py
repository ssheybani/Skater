import pandas as pd
import numpy as np
import unittest
import sys

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model, model_from_yaml
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from skater.util.image_ops import load_image, show_image, normalize, add_noise
from skater.core.local_interpretation.dnni.deep_interpreter import DeepInterpreter
from skater.core.visualizer.image_visualizer import visualize

class TestDNNI(unittest.TestCase):

    def build_model(self):
        # Build and train a network.
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                  verbose=1, validation_data=(self.x_test, self.y_test))


    def setUp(self):
        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 3
        # input image dimensions
        self.img_rows, self.img_cols = 28, 28
        # shuffled and split between train and test sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        self.x_train = (self.x_train - 0.5) * 2
        self.x_test = (self.x_test - 0.5) * 2

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)




if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRuleList)
    unittest.TextTestRunner(verbosity=2).run(suite)

