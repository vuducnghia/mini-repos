# https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model
from keras.utils import plot_model


def vgg_block(layer_in, n_filter, n_conv):
    for _ in range(n_conv):
        layer_in = Conv2D(filters=n_filter, kernel_size=(3, 3), padding='same', activation='relu')(layer_in)
    layer_in = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer_in)
    return layer_in


model_input = Input(shape=(256, 256, 3))
layer = vgg_block(layer_in=model_input, n_filter=64, n_conv=2)
layer = vgg_block(layer_in=layer, n_filter=128, n_conv=2)
layer = vgg_block(layer_in=layer, n_filter=256, n_conv=4)

model = Model(input=model_input, output=layer)
model.summary()
plot_model(model, show_shapes=True, to_file='vgg_block.png')
