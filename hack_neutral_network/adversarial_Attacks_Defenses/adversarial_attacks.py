# https://medium.com/analytics-vidhya/implementing-adversarial-attacks-and-defenses-in-keras-tensorflow-2-0-cab6120c5715
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--create_model', help='create_model', default=False)
args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

img_rows, img_cols, channels = 28, 28, 1
num_classes = 10

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print("Data shapes", x_test.shape, y_test.shape, x_train.shape, y_train.shape)


def create_model():
    model_input = Input(shape=(28, 28, 1))
    layer = Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(3, 3), activation='relu')(model_input)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(3, 3), activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(rate=0.2)(layer)
    layer = Flatten()(layer)
    layer = Dense(units=32, activation='relu')(layer)
    layer = Dropout(rate=0.2)(layer)
    layer = Dense(units=num_classes)(layer)
    layer = Activation(activation='softmax')(layer)

    model = Model(model_input, layer)
    opt = Adam(learning_rate=0.0005, beta_1=0.5)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    model.summary()

    return model


def adversarial_pattern(image, label, model):
    image = tf.cast(image, dtype=tf.float32)
    image = tf.Variable(image)
    with tf.GradientTape() as gtape:
        gtape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(y_true=label, y_pred=prediction)

    gradient = gtape.gradient(loss, image)
    signed_grad = tf.sign(gradient)

    return signed_grad


def generate_adversarials(batch_size, model):
    while True:
        x, y = [], []

        for batch in range(batch_size):
            N = random.randint(0, 100)
            label = y_train[N]
            image = x_train[N]

            perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), label, model).numpy()
            epsilon = 0.1
            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(label)

        x = np.array(x).reshape((batch_size, img_rows, img_cols, channels))
        y = np.array(y)

        yield x, y


if __name__ == '__main__':
    # training model
    if args.create_model:
        model = create_model()
        model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))
        model.save('model_mnist.h5')
    else:
        model = tf.keras.models.load_model('model_mnist.h5')

    image = x_train[0]
    image_label = y_train[0]
    perturbations = adversarial_pattern(image.reshape(1, img_rows, img_cols, channels), image_label, model)

    adversarial = image + perturbations * 0.1
    print(labels[model.predict(image.reshape(1, img_rows, img_cols, channels)).argmax()])
    print(labels[model.predict(adversarial).argmax()])

    if channels == 1:
        plt.imshow(tf.reshape(adversarial, shape=(img_rows, img_cols)))
    plt.show()

    x_adversarial_train, y_adversarial_train = next(generate_adversarials(20000, model))
    x_adversarial_test, y_adversarial_test = next(generate_adversarials(10000, model))

    print(f'Base accuracy on adversarial images: {model.evaluate(x=x_adversarial_test, y=y_test, verbose=0)}')

    model.fit(x_adversarial_train, y_adversarial_train, batch_size=128, epochs=5,
              validation_data=(x_adversarial_test, y_adversarial_test))

    print("Defended accuracy on adversarial images:",
          model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

    print("Defended accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))
