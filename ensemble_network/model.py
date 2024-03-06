# https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
# https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/
import os
from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, BatchNormalization, add, Input
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.utils import plot_model

learning_rate = 3e-3


def model1():
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(10, 224, 224, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    opt = SGD(lr=learning_rate, momentum=0.9, decay=1e-6)
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()
    # plot_model(model, show_shapes=True, to_file='report/model_graph_1.png')

    return model, 'model_1'


model1()


def model_leaky_relu():
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(10, 224, 224, 3), padding="same"))
    model.add(LeakyReLU())
    model.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    opt = SGD(lr=learning_rate, momentum=0.9, decay=1e-6)
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True, to_file='report/model_graph_leaky_relu.png')

    return model, 'model_leaky'


def stack_block2(layer_in, n_filter):
    conv1 = Conv3D(filters=n_filter, kernel_size=(3, 3, 3), padding='same')(layer_in)
    conv1 = LeakyReLU()(conv1)

    conv2 = Conv3D(filters=n_filter, kernel_size=(3, 3, 3), padding='same')(conv1)
    conv2 = LeakyReLU()(conv2)

    return conv2


def model_2():
    model_input = Input(shape=(10, 224, 224, 3))

    layer = stack_block2(model_input, 32)

    layer = MaxPooling3D(pool_size=(1, 3, 3))(layer)

    layer = stack_block2(layer, 64)

    layer = MaxPooling3D(pool_size=(1, 3, 3))(layer)

    layer = stack_block2(layer, 128)

    layer = MaxPooling3D(pool_size=(1, 3, 3))(layer)

    layer = stack_block2(layer, 256)
    layer = MaxPooling3D(pool_size=(1, 3, 3))(layer)

    layer = BatchNormalization()(layer)
    layer = Dropout(rate=0.25)(layer)

    layer = Flatten()(layer)
    layer = Dense(units=512)(layer)
    layer = Dense(units=7)(layer)

    model = Model(inputs=model_input, outputs=layer)

    opt = SGD(lr=learning_rate, momentum=0.9, decay=1e-6)
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True, to_file='report/model_graph_2.png')

    return model, 'model_2'


# model_2()


def residual_module(layer_in, n_filter, kernel_size=(1, 3, 3)):
    merge_input = layer_in
    if layer_in.shape[-1] != n_filter:
        merge_input = Conv3D(filters=4 * n_filter, kernel_size=(1, 1, 1), padding='same', activation='relu',
                             kernel_initializer='he_normal')(layer_in)
        merge_input = BatchNormalization()(merge_input)

    conv1 = Conv3D(filters=n_filter, kernel_size=(3, 1, 1), padding='same', activation='relu')(layer_in)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv3D(filters=n_filter, kernel_size=kernel_size, padding='same', activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv3D(filters=4 * n_filter, kernel_size=(1, 1, 1), padding='same', activation='relu')(conv2)
    conv3 = BatchNormalization()(conv3)

    layer_out = add([conv3, merge_input])
    layer_out = Activation('relu')(layer_out)

    return layer_out


def stack_block(layer, n_filter=32):
    layer = residual_module(layer, n_filter=n_filter, kernel_size=(1, 3, 3))
    layer = residual_module(layer, n_filter=n_filter, kernel_size=(1, 3, 3))
    layer = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(layer)
    return layer


def model_residual():
    model_input = Input(shape=(10, 224, 224, 3))
    conv1 = Conv3D(64, kernel_size=(1, 5, 5), input_shape=(10, 224, 224, 3), padding='same', strides=(1, 2, 2),
                   activation='relu')(model_input)
    layer = BatchNormalization()(conv1)
    layer = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(layer)

    layer = stack_block(layer, 32)
    layer = MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1))(layer)

    layer = stack_block(layer, 64)
    layer = MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1))(layer)

    layer = stack_block(layer, 128)
    layer = MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1))(layer)

    layer = stack_block(layer, 256)

    # layer = GlobalAveragePooling3D()(layer)
    # layer = MaxPooling3D(pool_size=(3, 3, 3))(layer)

    layer = Flatten()(layer)
    layer = Dense(units=512)(layer)
    layer = Dense(units=7)(layer)

    model = Model(inputs=model_input, outputs=layer)

    opt = SGD(lr=learning_rate, momentum=0.9, decay=1e-6)
    model.compile(loss=categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True, to_file='report/model_graph_residual.png')

    return model, 'model_residual'


# model_residual()


def load_all_models(folder_model='models'):
    all_models = list()
    index = 0
    for file in os.listdir(folder_model):
        print(f'----Load model: {file}')
        model = load_model(f'{folder_model}/{file}')
        for layer in model.layers:
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(index + 1) + '_' + layer.name

        model.save(f'rename_models/model_{index}.h5')

        index += 1

    for file in os.listdir('rename_models'):
        model = load_model(f'rename_models/{file}')
        all_models.append(model)

    return all_models


def ensemble():
    models = load_all_models('models')

    ensemble_input = [model.input for model in models]
    ensemble_outputs = [model.output for model in models]
    merge = concatenate(ensemble_outputs)

    hidden = Dense(units=256, activation='relu')(merge)
    output = Dense(units=7, activation='softmax')(hidden)

    model = Model(inputs=ensemble_input, outputs=output)

    opt = SGD(lr=learning_rate, momentum=0.9, decay=1e-6)
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    plot_model(model, show_shapes=True, to_file='report/ensemble.png')

    # model.save('ensemble.h5')
    model.summary()
    return model, 'model_ensemble'


# ensemble()
