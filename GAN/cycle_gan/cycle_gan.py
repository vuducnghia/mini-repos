# https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
# https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
# https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
# https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array, load_img
from keras.initializers import RandomNormal
from keras.layers import Input, Conv2D, LeakyReLU, Activation, Concatenate, Conv2DTranspose
from keras.utils.vis_utils import plot_model
from keras.models import Model, load_model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np
import os
from random import random
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class CycleGan:
    def __init__(self):
        testA = self.load_data('horse2zebra/testA/')
        testB = self.load_data('horse2zebra/testB/')
        # trainA = self.load_data('horse2zebra/trainA/')
        # trainB = self.load_data('horse2zebra/trainB/')
        # self.dataA = np.vstack((testA, trainA))
        # self.dataB = np.vstack((testB, trainB))

        self.dataA = testA
        self.dataB = testB
        print('------------------------')
        print(self.dataA.shape)
        print(self.dataB.shape)

    def load_data(self, path, size=(256, 256)):
        datalist = list()
        for filename in os.listdir(path):
            pixels = load_img(path + filename, target_size=size)
            pixels = img_to_array(pixels)
            # should use standardization prefer normalization
            # in this case, i assume mean is 127,5
            pixels = (pixels - 127.5) / 127.5
            datalist.append(pixels)

        return np.asarray(datalist)

    def generate_real_samples(self, dataset, n_samples, patch_shape):
        ix = np.random.randint(low=0, high=dataset.shape[0], size=n_samples)

        X = dataset[ix]
        y = np.ones(shape=(n_samples, patch_shape, patch_shape, 1))

        return X, y

    def generate_fake_samples(self, g_model, dataset, patch_shape):
        X = g_model.predict(dataset)
        y = np.zeros(shape=(len(X), patch_shape, patch_shape, 1))

        return X, y

    def define_discriminator(self, image_shape):
        # weight initialization
        init_weight = RandomNormal(stddev=0.02)

        model_input = Input(shape=image_shape)
        layer = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init_weight)(
            model_input)
        layer = LeakyReLU(alpha=0.2)(layer)

        layer = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init_weight)(
            layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)

        layer = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init_weight)(
            layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)

        layer = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init_weight)(
            layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)

        layer = Conv2D(filters=512, kernel_size=(4, 4), padding='same', kernel_initializer=init_weight)(
            layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)

        patch_out = Conv2D(filters=1, kernel_size=(4, 4), padding='same', kernel_initializer=init_weight)(layer)

        model_discriminator = Model(model_input, patch_out)
        model_discriminator.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        # model_discriminator.summary()
        plot_model(model_discriminator, to_file='discriminator_model.png', show_shapes=True)

        return model_discriminator

    def resnet_block(self, n_filter, input_layer):
        init_weight = RandomNormal(stddev=0.02)

        layer = Conv2D(n_filter, kernel_size=(3, 3), padding='same', kernel_initializer=init_weight)(input_layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(n_filter, kernel_size=(3, 3), padding='same', kernel_initializer=init_weight)(layer)
        layer = InstanceNormalization(axis=-1)(layer)

        layer = Concatenate()([layer, input_layer])

        return layer

    def define_generator(self, image_shape, n_resnet=9):
        init_weight = RandomNormal(stddev=0.02)
        model_input = Input(shape=image_shape)

        layer = Conv2D(64, kernel_size=(7, 7), padding='same', kernel_initializer=init_weight)(model_input)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=init_weight)(layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=init_weight)(layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = Activation('relu')(layer)

        for _ in range(n_resnet):
            layer = self.resnet_block(256, layer)

        layer = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                kernel_initializer=init_weight)(layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = Activation('relu')(layer)

        layer = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                kernel_initializer=init_weight)(layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(3, kernel_size=(7, 7), padding='same', kernel_initializer=init_weight)(layer)
        layer = InstanceNormalization(axis=-1)(layer)
        layer = Activation('tanh')(layer)

        model_generator = Model(inputs=model_input, outputs=layer)
        # model_generator.summary()
        plot_model(model_generator, to_file='generator_model.png', show_shapes=True)

        return model_generator

    def define_composite_model(self, g_model_1, d_model, g_model_2, image_shape):
        g_model_1.trainable = True
        g_model_2.trainable = False
        d_model.trainable = False

        # Adversarial Loss: Domain-B -> Generator-A -> Domain-A -> Discriminator-A -> [real/fake]
        # image domain B
        input_gen = Input(shape=image_shape)
        gen1_out = g_model_1(input_gen)
        output_discriminator = d_model(gen1_out)

        # Identity Loss: Domain-A -> Generator-A -> Domain-A
        # image domain A
        input_identity = Input(shape=image_shape)
        output_identity = g_model_1(input_identity)

        # Forward Cycle Loss: Domain B -> Generator-A -> Domain-A -> Generator-B -> Domain-B
        output_forward = g_model_2(gen1_out)

        # Backward Cycle Loss: Domain-A -> Generator-B -> Domain-B -> Generator-A -> Domain-A
        gen2_out = g_model_2(input_identity)
        output_backward = g_model_1(gen2_out)

        # inputs model: Domain-B, Domain-A
        model_composite = Model([input_gen, input_identity],
                                [output_discriminator, output_identity, output_forward, output_backward])
        opt = Adam(lr=0.0002, beta_1=0.5)
        model_composite.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
        model_composite.summary()
        plot_model(model_composite, to_file='model_composite.png', show_shapes=True)

        return model_composite

    def save_models(self, step, g_model_1, g_model_2):
        filename_1 = f'g_model_1_{step}.h5'
        filename_2 = f'g_model_2_{step}.h5'
        g_model_1.save(filename_1)
        g_model_2.save(filename_2)

    def summarize_performance(self, dataset, step, g_model, name, n_samples=5):
        # select a sample of input image
        X_in, _ = self.generate_real_samples(dataset, n_samples, 0)

        # generate translated images
        X_out, _ = self.generate_fake_samples(g_model, X_in, 0)

        # scale all pixels from [-1, 1] to [0, 1]
        X_in = (X_in + 1) / 2.0
        X_out = (X_out + 1) / 2.0

        for i in range(n_samples):
            plt.subplot(2, n_samples, i + 1)
            plt.axis('off')
            plt.imshow(X_in[i])

        for i in range(n_samples):
            plt.subplot(2, n_samples, i + 1 + n_samples)
            plt.axis('off')
            plt.imshow(X_out[i])

        plt.savefig(f'{name}generated_{step}.png')
        plt.close()

    # update the discriminators using a history of generated images rather than the ones produced by the latest generators
    def update_image_pool(self, images, max_size=50):
        selected, pool = list(), list()

        for image in images:
            if len(pool) < max_size:
                pool.append(image)
                selected.append(image)
            elif random() < 0.5:
                selected.append(image)
            else:
                ix = np.random.randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image

        return np.asarray(selected)

    def train(self, d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA):
        n_epochs, n_batch = 100, 1
        n_patch = d_model_A.output_shape[1]

        batch_per_epoch = len(self.dataA)
        n_steps = batch_per_epoch * n_epochs

        for i in range(n_steps):
            X_real_A, y_real_A = self.generate_real_samples(self.dataA, n_batch, n_patch)
            X_real_B, y_real_B = self.generate_real_samples(self.dataB, n_batch, n_patch)

            X_fake_A, y_fake_A = self.generate_fake_samples(g_model_BtoA, X_real_B, n_patch)
            X_fake_B, y_fake_B = self.generate_fake_samples(g_model_AtoB, X_real_A, n_patch)

            X_fake_A = self.update_image_pool(X_fake_A)
            X_fake_B = self.update_image_pool(X_fake_B)

            # update generator B to A via adversarial and cycle loss
            g_loss_2, _, _, _, _ = c_model_BtoA.train_on_batch(x=[X_real_B, X_real_A],
                                                               y=[y_real_A, X_real_A, X_real_B, X_real_A])

            # update discriminator for A -> [real/fake]
            dA_loss_1 = d_model_A.train_on_batch(X_real_A, y_real_A)
            dA_loss_2 = d_model_A.train_on_batch(X_fake_A, y_fake_A)

            # update generator A to B via adversarial and cycle loss
            g_loss_1, _, _, _, _ = c_model_AtoB.train_on_batch(x=[X_real_A, X_real_B],
                                                               y=[y_real_B, X_real_B, X_real_A, X_real_B])

            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_real_B, y_real_B)
            dB_loss2 = d_model_B.train_on_batch(X_fake_B, y_fake_B)

            print(
                f'step: {i + 1}, dA_loss: [{dA_loss_1}, {dA_loss_2}], dB_loss: [{dB_loss1}, {dB_loss2}], g_loss: [{g_loss_1}, {g_loss_2}]')

            if (i + 1) % batch_per_epoch == 0:
                # plot A->B translation
                self.summarize_performance(self.dataA, i, g_model_AtoB, 'AtoB')

                # plot B->A translation
                self.summarize_performance(self.dataB, i, g_model_BtoA, 'BtoA')

            if (i + 1) % (batch_per_epoch * 5) == 0:
                self.save_models(i, g_model_AtoB, g_model_BtoA)


def predict(g_model, filename, size=(256, 256)):
    pixels = load_img(filename, target_size=size)
    pixels = img_to_array(pixels)
    pixels = np.expand_dims(pixels, 0)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5

    image_tar = g_model.predict(pixels)
    image_tar = (image_tar + 1) / 2.0
    # plot the translated image
    plt.imshow(image_tar[0])
    plt.show()


if __name__ == '__main__':
    # CG = CycleGan()
    #
    # image_shape = (256, 256, 3)
    #
    # g_model_AtoB = CG.define_generator(image_shape)
    # g_model_BtoA = CG.define_generator(image_shape)
    #
    # d_model_A = CG.define_discriminator(image_shape)
    # d_model_B = CG.define_discriminator(image_shape)
    #
    # c_model_AtoB = CG.define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    # c_model_BtoA = CG.define_composite_model(g_model_BtoA, d_model_B, g_model_AtoB, image_shape)
    #
    # CG.train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA)

    # predict
    cust = {'InstanceNormalization': InstanceNormalization}
    model = load_model('g_model_1_77154.h5', cust)
    predict(model, 'horse2zebra/testA/n02381460_200.jpg')
