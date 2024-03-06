from keras.layers import Conv2D, MaxPooling2D, add, Activation, Input
from keras.models import Model
from keras.utils.vis_utils import plot_model


# from k
def residual_module(layer_in, n_filter):
    merge_input = layer_in
    if layer_in.shape[-1] != n_filter:
        merge_input = Conv2D(filters=n_filter, kernel_size=(1, 1), padding='same', activation='relu',
                             kernel_initializer='he_normal')(layer_in)
    conv1 = Conv2D(filters=n_filter, kernel_size=(3, 3), padding='same', activation='relu',
                   kernel_initializer='he_normal')(layer_in)
    conv2 = Conv2D(filters=n_filter, kernel_size=(3, 3), padding='same', activation='relu',
                   kernel_initializer='he_normal')(conv1)

    layer_out = add([conv2, merge_input])
    layer_out = Activation('relu')(layer_out)

    return layer_out


model_input = Input(shape=(256, 256, 3))
layer = residual_module(model_input, 64)
model = Model(inputs=model_input, outputs=layer)
model.summary()
plot_model(model, to_file='residual.png', show_shapes=True)