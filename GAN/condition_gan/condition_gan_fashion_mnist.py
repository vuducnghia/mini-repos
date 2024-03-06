# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, LeakyReLU, Reshape, Conv2DTranspose, Embedding, \
    Concatenate
from keras.models import Model, Sequential
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model


class C_GAN:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.n_classes = 10
        self.load_real_data()
        self.define_discriminator()
        self.define_generator()
        self.define_cgan()

    def load_real_data(self):
        (self.x_train, self.y_train), (_, _) = fashion_mnist.load_data()
        # expand to 3D
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_train.astype('float32')
        # scale from [0,255] to [-1,1]
        self.x_train = (self.x_train - 127.5) / 127.5

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = np.random.randint(low=0, high=self.x_train.shape[0], size=n_samples)
        X, labels = self.x_train[ix], self.y_train[ix]
        y = np.ones(shape=(n_samples, 1))

        return [X, labels], y

    def generate_fake_samples(self, n_samples):
        # generate points in latent space
        z_input, labels_input = self.generate_latent_points(n_samples)
        # predict outputs
        images = self.model_generator.predict([z_input, labels_input])
        y = np.zeros((n_samples, 1))

        return [images, labels_input], y

    def generate_latent_points(self, n_samples):
        x_input = np.random.randn(self.latent_dim * n_samples)
        z_input = x_input.reshape(n_samples, self.latent_dim)
        labels = np.random.randint(0, self.n_classes, size=n_samples)

        return [z_input, labels]

    def define_discriminator(self):
        input_label = Input(shape=(1,))
        layer_label = Embedding(input_dim=self.n_classes, output_dim=50)(input_label)
        layer_label = Dense(units=28 * 28)(layer_label)
        layer_label = Reshape(target_shape=(28, 28, 1))(layer_label)

        input_image = Input(shape=(28, 28, 1))
        merge = Concatenate()([input_image, layer_label])

        layer = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(merge)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Flatten()(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(units=1, activation='sigmoid')(layer)

        self.model_discriminator = Model(inputs=[input_label, input_image], outputs=layer)
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.model_discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model_discriminator.summary()
        plot_model(self.model_discriminator, to_file='discriminator_model.png', show_shapes=True)

    def define_generator(self):
        input_label = Input(shape=(1,))
        layer_label = Embedding(input_dim=self.n_classes, output_dim=50)(input_label)
        layer_label = Dense(units=7 * 7)(layer_label)
        layer_label = Reshape(target_shape=(7, 7, 1))(layer_label)

        input_latent = Input(shape=(self.latent_dim,))
        layer_latent = Dense(units=128 * 7 * 7)(input_latent)
        layer_latent = LeakyReLU(alpha=0.2)(layer_latent)
        layer_latent = Reshape(target_shape=(7, 7, 128))(layer_latent)

        merge = Concatenate()([layer_latent, layer_label])
        layer = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2D(1, kernel_size=(7, 7), activation='tanh', padding='same')(layer)

        self.model_generator = Model(inputs=[input_latent, input_label], outputs=layer)
        self.model_generator.summary()
        plot_model(self.model_generator, to_file='generate_model.png', show_shapes=True)

    def define_cgan(self):
        model_discriminator = self.model_discriminator
        model_discriminator.trainable = False
        gen_noise, gen_label = self.model_generator.input
        # get image from generator model
        gen_output = self.model_generator.output
        print(gen_noise, gen_label)
        # collect image output and label output from generator as input to discriminator
        cgan_output = model_discriminator([gen_label, gen_output])

        self.model_cgan = Model(inputs=[gen_noise, gen_label], outputs=cgan_output)
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.model_cgan.compile(loss='binary_crossentropy', optimizer=opt)
        self.model_cgan.summary()
        plot_model(self.model_cgan, to_file='cgan_model.png', show_shapes=True)

    def train(self, n_epochs=100, n_batch=256):
        half_batch = int(n_batch / 2)
        bat_per_epo = int(self.x_train.shape[0] / n_batch)
        for i in range(n_epochs):
            for j in range(bat_per_epo):
                [X_real, labels_real], y_real = self.generate_real_samples(half_batch)
                [X_fake, labels_fake], y_fake = self.generate_fake_samples(half_batch)

                d_loss1, _ = self.model_discriminator.train_on_batch([labels_real, X_real], y_real)
                d_loss2, _ = self.model_discriminator.train_on_batch([labels_fake, X_fake], y_fake)

                z_input, labels = self.generate_latent_points(n_batch)
                y_cgan = np.ones((n_batch, 1))
                g_loss = self.model_cgan.train_on_batch([z_input, labels], y_cgan)

                print(f'epoch:{i + 1}, {j + 1}/{bat_per_epo}, d_loss1: {d_loss1}, d_loss2: {d_loss2}, g_loss: {g_loss}')

            if (i + 1) % 10 == 0:
                self.summarize_performance(i)

    def summarize_performance(self, epoch, n_samples=100):
        [X_real, labels_real], y_real = self.generate_real_samples(n_samples)
        [X_fake, labels_fake], y_fake = self.generate_fake_samples(n_samples)

        _, acc_real = self.model_discriminator.evaluate([labels_real, X_real], y_real)
        _, acc_fake = self.model_discriminator.evaluate([labels_fake, X_fake], y_fake)

        print(f'Accuracy real: {acc_real}, fake: {acc_fake}')
        self.save_plot(X_fake, epoch)

    def save_model(self):
        self.model_generator.save('generator.h5')

    def save_plot(self, examples, epoch, n=10):
        for i in range(n * n):
            # define subplot
            plt.subplot(n, n, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i, :, :, 0], cmap='gray_r')
            # save plot to file
        filename = 'generated_plot_e%03d.png' % (epoch + 1)
        plt.savefig(filename)
        plt.close()


if __name__ == '__main__':
    cG = C_GAN(100)
    cG.train()
