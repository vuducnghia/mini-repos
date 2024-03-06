from tensorflow.keras.layers import Conv2D, Activation, Input, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, Model


class teacher_model:
    def __init__(self):
        super().__init__()

    @staticmethod
    def build_model():
        input = Input(shape=(28, 28, 1))
        layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(input)
        layer = Conv2D(64, kernel_size=(3, 3), activation='relu')(layer)
        layer = MaxPool2D(pool_size=(2, 2))(layer)
        layer = Dropout(rate=0.25)(layer)
        layer = Flatten()(layer)
        layer = Dense(128, activation='relu')(layer)
        layer = Dense(10)(layer)
        layer = Activation('softmax')(layer)

        return Model(inputs=input, outputs=layer)

class student_model:
    def __init__(self):
        super().__init__()

    @staticmethod
    def build_model():
        input = Input(shape=(28, 28, 1))
        layer = Flatten()(input)
        layer = Dense(32, activation='relu')(layer)
        layer = Dropout(rate=0.25)(layer)
        layer = Dense(10)(layer)
        layer = Activation('softmax')(layer)

        return Model(inputs=input, outputs=layer)