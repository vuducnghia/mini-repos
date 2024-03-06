from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model



def naive_inception_module(layer_in, filter1, filter2, filter3):
    conv1 = Conv2D(filters=filter1, kernel_size=(1, 1), padding='same', activation='relu')(layer_in)
    conv3 = Conv2D(filters=filter2, kernel_size=(3, 3), padding='same', activation='relu')(layer_in)
    conv5 = Conv2D(filters=filter3, kernel_size=(5, 5), padding='same', activation='relu')(layer_in)
    pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(layer_in)

    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)

    return layer_out


model_input = Input(shape=(256, 256, 3))
layer = naive_inception_module(model_input, 64, 128, 32)
model = Model(inputs=model_input, outputs=layer)

model.summary()
plot_model(model, to_file='native_inception.png', show_shapes=True)

def inception_module(layer_in, filter1, filter2_in, filter2_out, filter3_in, filter3_out, filter4_out):
    conv1 = Conv2D(filters=filter1, kernel_size=(1, 1), padding='same', activation='relu')(layer_in)

    conv3 = Conv2D(filters=filter2_in, kernel_size=(1, 1), padding='same', activation='relu')(layer_in)
    conv3 = Conv2D(filters=filter2_out, kernel_size=(3, 3), padding='same', activation='relu')(conv3)

    conv5 = Conv2D(filters=filter3_in, kernel_size=(1, 1), padding='same', activation='relu')(layer_in)
    conv5 = Conv2D(filters=filter3_out, kernel_size=(5, 5), padding='same', activation='relu')(conv5)

    pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = Conv2D(filters=filter4_out, kernel_size=(1,1), padding='same', activation='relu')(pool)

    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)

    return layer_out

model_input = Input(shape=(256, 256, 3))
layer = inception_module(model_input, 64, 96, 128, 16, 32, 32)
layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
model = Model(inputs=model_input, outputs=layer)

model.summary()
plot_model(model, to_file='inception.png', show_shapes=True)