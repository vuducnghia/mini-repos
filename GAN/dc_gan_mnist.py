# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
import cv2
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, LeakyReLU, Reshape, Conv2DTranspose
from keras.models import Model, Sequential
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam


class GAN:
    def __init__(self):
        self.load_real_data()
        self.define_discriminator()
        self.define_generator(100)

    def load_real_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # expand to 3D
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_train.astype('float32')
        self.x_train = self.x_train / 255.0

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = np.random.randint(low=0, high=self.x_train.shape[0], size=n_samples)
        X = self.x_train[ix]
        y = np.ones(shape=(n_samples, 1))

        return X, y

    def generate_fake_samples(self, latent_dim, n_samples):
        # generate points in latent space
        x_input = self.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        X = self.model_generator.predict(x_input)
        y = np.zeros((n_samples, 1))

        return X, y

    def define_discriminator(self):
        model_input = Input(shape=(28, 28, 1))
        layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(model_input)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dropout(rate=0.4)(layer)

        layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dropout(rate=0.4)(layer)

        layer = Flatten()(layer)
        layer = Dense(units=1, activation='sigmoid')(layer)

        self.model_discriminator = Model(inputs=model_input, outputs=layer)
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.model_discriminator.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # half batch for real and remain for fake
    def train_discriminator(self, n_iter=100, n_batch=256):
        half_batch = int(n_batch / 2)
        for i in range(n_iter):
            X_real, y_real = self.generate_real_samples(half_batch)
            X_fake, y_fake = self.generate_fake_samples(half_batch)
            # update discriminator on real samples
            _, real_acc = self.model_discriminator.train_on_batch(x=X_real, y=y_real)
            # update discriminator on fake samples
            _, fake_acc = self.model_discriminator.train_on_batch(x=X_fake, y=y_fake)

            print(f'iter: {i + 1}, real: {real_acc}, fake: {fake_acc}')

    # ------------------------------    generator   --------------------------

    def define_generator(self, latent_dim):
        # foundation for 7*7 image
        n_nodes = 128 * 7 * 7
        model_input = Input(shape=(latent_dim,))
        layer = Dense(units=n_nodes)(model_input)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Reshape(target_shape=(7, 7, 128))(layer)
        # upsample to 14*14
        layer = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        # upsample to 28*28
        layer = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Conv2D(filters=1, kernel_size=(7, 7), activation='sigmoid', padding='same')(layer)

        self.model_generator = Model(inputs=model_input, outputs=layer)
        self.model_generator.summary()

    def generate_latent_points(self, latent_dim, n_samples):
        x_input = np.random.randn(latent_dim * n_samples)
        x_input = x_input.reshape(n_samples, latent_dim)

        return x_input

    def define_gan(self, discriminator_model, generate_model):
        # make weights in discriminator not trainable
        discriminator_model.trainable = False
        self.model_gan = Sequential()
        self.model_gan.add(generate_model)
        self.model_gan.add(discriminator_model)
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.model_gan.compile(loss='binary_crossentropy', optimizer=opt)
        self.model_gan.summary()

    def summarize_performance(self, epoch, latent_dim, n_samples=100):
        X_real, y_real = self.generate_real_samples(n_samples)
        X_fake, y_fake = self.generate_fake_samples(latent_dim, n_samples)

        _, acc_real = self.model_discriminator.evaluate(X_real, y_real)
        _, acc_fake = self.model_discriminator.evaluate(X_fake, y_fake)

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

    def train_gan(self, latent_dim, n_epochs=100, n_batch=256):
        self.define_gan(self.model_discriminator, self.model_generator)
        half_batch = int(n_batch / 2)
        bat_per_epo = int(self.x_train.shape[0] / n_batch)

        for i in range(n_epochs):
            for j in range(bat_per_epo):
                X_real, y_real = self.generate_real_samples(half_batch)
                X_fake, y_fake = self.generate_fake_samples(latent_dim, half_batch)
                X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

                # update discriminator model weights
                d_loss, _ = self.model_discriminator.train_on_batch(X, y)

                # prepare point in latent space as input for the generator
                X_gan = self.generate_latent_points(latent_dim, n_batch)
                y_gan = np.ones(shape=(n_batch, 1))

                # update generator via disciminator 's error
                g_loss = self.model_gan.train_on_batch(X_gan, y_gan)

                print(f'epoch:{i + 1}, {j+1}/{bat_per_epo} d_loss: {d_loss}, g_loss: {g_loss}')

            if (i + 1) % 10 == 0:
                self.summarize_performance(i, latent_dim)

        self.save_model()


if __name__ == '__main__':
    g = GAN()

    g.train_gan(100)
