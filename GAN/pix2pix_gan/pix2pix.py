# https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.initializers import RandomNormal
from keras.layers import Input, Concatenate, Conv2D, LeakyReLU, BatchNormalization, Activation, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import Adam
import os
from matplotlib import pyplot as plt


def load_images(path, size=(256, 512)):
    src_list, tar_list = [], []

    for filename in os.listdir(path):
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)

        # split into satellite and map
        sat_img, map_img = pixels[:, :256], pixels[:, 256:]

        src_list.append(sat_img)
        tar_list.append(map_img)

    return np.asarray(src_list), np.asarray(tar_list)


def define_discriminator(image_shape):
    init_weight = RandomNormal(stddev=0.02)
    input_src_image = Input(shape=image_shape)
    input_target_image = Input(shape=image_shape)

    input_model = Concatenate()([input_src_image, input_target_image])
    layer = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=init_weight)(input_model)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=init_weight)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=init_weight)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=init_weight)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(filters=512, kernel_size=(4, 4), padding='same', kernel_initializer=init_weight)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(filters=1, kernel_size=(4, 4), padding='same', kernel_initializer=init_weight)(layer)
    layer = Activation(activation='sigmoid')(layer)

    model = Model([input_src_image, input_target_image], layer)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', loss_weights=[0.5])

    return model


def encode_block(layer_in, n_filters, batchnorm=True):
    init_weight = RandomNormal(stddev=0.02)

    layer = Conv2D(n_filters, kernel_size=(4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=init_weight)(layer_in)
    if batchnorm:
        layer = BatchNormalization()(layer, training=True)

    layer = LeakyReLU(alpha=0.2)(layer)

    return layer


def decode_block(layer_in, skip_in, n_filters, dropout=True):
    init_weight = RandomNormal(stddev=0.02)

    layer = Conv2DTranspose(filters=n_filters, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer=init_weight)(layer_in)
    layer = BatchNormalization()(layer, training=True)

    if dropout:
        layer = Dropout(rate=0.5)(layer, training=True)

    # merge with skip connection
    layer = Concatenate()([layer, skip_in])
    layer = Activation(activation='relu')(layer)

    return layer


def define_generator(image_shape=(256, 256, 3)):
    init_weight = RandomNormal(stddev=0.02)

    input_model = Input(shape=image_shape)

    # encode model
    e1 = encode_block(input_model, n_filters=64, batchnorm=False)
    e2 = encode_block(e1, n_filters=128)
    e3 = encode_block(e2, n_filters=256)
    e4 = encode_block(e3, n_filters=512)
    e5 = encode_block(e4, n_filters=512)
    e6 = encode_block(e5, n_filters=512)
    e7 = encode_block(e6, n_filters=512)

    # bottleneck, no batchnormalize and relu
    b = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init_weight)(e7)
    b = Activation('relu')(b)

    # decode model
    d1 = decode_block(layer_in=b, skip_in=e7, n_filters=512)
    d2 = decode_block(layer_in=d1, skip_in=e6, n_filters=512)
    d3 = decode_block(layer_in=d2, skip_in=e5, n_filters=512)
    d4 = decode_block(layer_in=d3, skip_in=e4, n_filters=512, dropout=False)
    d5 = decode_block(layer_in=d4, skip_in=e3, n_filters=256, dropout=False)
    d6 = decode_block(layer_in=d5, skip_in=e2, n_filters=128, dropout=False)
    d7 = decode_block(layer_in=d6, skip_in=e1, n_filters=64, dropout=False)

    # output
    layer = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init_weight)(d7)

    # ensure pixels output in range [-1, 1]
    layer = Activation('tanh')(layer)

    model = Model(input_model, layer)

    return model


def define_gan(g_model, d_model, image_shape):
    d_model.trainable = False

    input_src_image = Input(shape=image_shape)

    gen_out = g_model(input_src_image)
    dis_out = d_model([input_src_image, gen_out])

    model = Model(input_src_image, [dis_out, gen_out])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss=['binary_crossentropy', 'mae'], loss_weights=[1, 100])

    return model


def load_real_samples(filename):
    data = np.load(filename)

    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5

    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = np.random.randint(low=0, high=trainA.shape[0], size=n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = np.ones(shape=(n_samples, patch_shape, patch_shape, 1))

    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = np.zeros((len(samples), patch_shape, patch_shape, 1))

    return X, y


def summarize_performance(step, g_model, dataset, n_samples=5):
    [X_real_A, X_real_B], _ = generate_real_samples(dataset, n_samples, 0)
    X_fake, _ = generate_fake_samples(g_model, X_real_A, 0)

    X_real_A = (X_real_A + 1) / 2.0
    X_real_B = (X_real_B + 1) / 2.0

    for i in range(n_samples):
        plt.subplot(3, n_samples, i + 1)
        plt.axis('off')
        plt.imshow(X_real_A[i])

    for i in range(n_samples):
        plt.subplot(3, n_samples, i + 1 + n_samples)
        plt.axis('off')
        plt.imshow(X_real_B[i])

    for i in range(n_samples):
        plt.subplot(3, n_samples, i + 1 + n_samples * 2)
        plt.axis('off')
        plt.imshow(X_fake[i])

    g_model.save(f'generated_{step}.h5')
    plt.savefig(f'generated_{step}.png')
    plt.close()


def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    n_patch = d_model.output_shape[1]

    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)


if __name__ == '__main__':
    [src_images, tar_images] = load_images('train/')
    print('Loaded: ', src_images.shape, tar_images.shape)
    # save as compressed numpy array
    filename = 'maps_256.npz'
    np.savez_compressed(filename, src_images, tar_images)

    dataset = load_real_samples('maps_256.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)
    # define input shape based on the loaded dataset
    image_shape = dataset[0].shape[1:]

    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    gan_model = define_gan(g_model, d_model, image_shape)

    train(d_model, g_model, gan_model, dataset)
