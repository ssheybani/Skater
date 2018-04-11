import keras
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model, model_from_yaml
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import unittest
from skater.core.local_interpretation.dnni.deep_interpreter import DeepInterpreter


class TestDNNI(unittest.TestCase):

    def _build_model(self):
        # Build and train a network.
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
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
        return model


    def _save(self):
        model_yaml = self.model.to_yaml()
        with open("sample_model.yaml", "w") as f_h:
            f_h.write(model_yaml)
        # serialize weights to HDF5
        self.model.save_weights("sample_model.h5")


    def _load(self):
        f_h = open('sample_model.yaml', 'r')
        persisted_yaml_model = f_h.read()
        f_h.close()
        loaded_model = model_from_yaml(persisted_yaml_model)
        # load weights into retrieved model instance
        loaded_model.load_weights("sample_model.h5")
        return loaded_model


    def setUp(self):
        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 2
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
            self.input_shape = (self.img_rows, self.img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        self.x_train = (self.x_train - 0.5) * 2
        self.x_test = (self.x_test - 0.5) * 2

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

        # build a simple CovNet model
        self.model = self.build_model()

        # save the model
        self._save()


    def test_deep_interpreter_cnn(self):
        K.set_learning_phase(0)
        with DeepInterpreter(session=K.get_session()) as di:
            # 1. Load the persisted model
            # 2. Retrieve the input tensor from the loaded model
            retrieved_model = self._load()
            input_tensor = retrieved_model.layers[0].input
            output_tensor = retrieved_model.layers[-2].output

            # 3. We will using the last dense layer(pre-softmax) as the output layer
            # 4. Instantiate a model with the new input and output tensor
            new_model = Model(inputs=input_tensor, outputs=output_tensor)
            target_tensor = new_model(input_tensor)
            xs = self.x_test[0:2]
            ys = self.y_test[0:2]

            relevance_scores = di.explain('elrp', target_tensor * ys, input_tensor, xs)
        self.assertEquals(relevance_scores.shape, (2, 28, 28, 1))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDNNI)
    unittest.TextTestRunner(verbosity=2).run(suite)
