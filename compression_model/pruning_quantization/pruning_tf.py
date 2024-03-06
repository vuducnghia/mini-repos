import tensorflow as tf
from keras.datasets import mnist
from keras import utils
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.models import Model, load_model
from keras.optimizers import SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class Model_Classify:
    def __init__(self):
        # shape x_train: (60000, 28, 28)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)

        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        self.y_train = utils.to_categorical(self.y_train, 10)
        self.y_test = utils.to_categorical(self.y_test, 10)
        print(self.x_train.shape, self.y_train.shape)
        # self.build_model()

    def build_model(self):
        model_input = Input(shape=(28, 28, 1))
        layer = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(model_input)
        layer = MaxPooling2D(pool_size=(2, 2))(layer)
        layer = BatchNormalization()(layer)

        layer = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(layer)
        layer = MaxPooling2D(pool_size=(2, 2))(layer)

        layer = Flatten()(layer)
        layer = Dense(units=1024, activation='relu')(layer)
        layer = Dense(units=10, activation='softmax')(layer)

        self.model = Model(inputs=model_input, outputs=layer)
        self.model.summary()

        opt = SGD(lr=0.001, momentum=0.9, decay=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return self.model

    def train(self):
        self.model.fit(x=self.x_train, y=self.y_train, batch_size=64, epochs=3,
                       validation_data=(self.x_test, self.y_test))
        self.model.save('model.h5')

    def load_model(self):
        self.model = load_model('model.h5')
        for layer in self.model.layers:
            print(layer)

if __name__ == '__main__':
    m = Model_Classify()
    # m.build_model()
    # m.train()
    m.load_model()
