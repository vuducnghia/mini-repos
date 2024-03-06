# https://towardsdatascience.com/knowledge-distillation-and-the-concept-of-dark-knowledge-8b7aed8014ac
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from model import student_model, teacher_model

n_classes = 10
epochs = 50
batch_size = 128
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# convert y_train and y_test to categorical binary values
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# nomallize image
X_train /= 255.0
X_test /= 255.0

opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.001)


def plot_loss(history, name):
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'model {name} accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    # train teacher model
    # t_model = teacher_model.build_model()
    # t_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # t_model.summary()
    #
    # history = t_model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
    # t_model.save('model/teacher_model.h5')
    # plot_loss(history, 'teacher')

    # train origin student model
    s_model = student_model.build_model()
    s_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    s_model.summary()

    history = s_model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
    s_model.save('model/teacher_model.h5')
    plot_loss(history, 'teacher')
