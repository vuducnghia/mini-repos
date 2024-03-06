from data_generator import Generator
from model import AttentionOCR
from vocabulary import Vocabulary
import time
import os
import tensorflow as tf
import argparse
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.001
embedding_dim = 256
vocab_size = 43 + 3
max_txt_length = 30 + 3
encode_units = 256
decode_units = 512
image_height = 32
image_width = 320

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--finetune", help="debug", default=False)
args = vars(ap.parse_args())

vocab = Vocabulary()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

Attention = AttentionOCR(image_width, image_height, vocab_size, encode_units, decode_units, embedding_dim,
                         max_txt_length, args['finetune'])
model_train = Attention.build_model_train()
model_train.summary()
model_valid = Attention.build_model_inference()


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# @tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the ocr are not related from image to image
    hidden = tf.zeros((BATCH_SIZE, decode_units))

    with tf.GradientTape() as tape:
        decode_input = tf.expand_dims([vocab.word_index('<START>')] * BATCH_SIZE, axis=1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target.shape[1]):
            predictions, hidden = model_train([img_tensor, decode_input, hidden])
            loss += loss_function(target[:, t], predictions)

            # using teacher forcing
            decode_input = tf.expand_dims(target[:, t], 1)

    batch_loss = (loss / int(target.shape[1]))
    variables = model_train.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# @tf.function
def validate_step(img_tensor, target):
    decode_input = tf.expand_dims([vocab.word_index('<START>')] * BATCH_SIZE, axis=1)
    hidden = tf.zeros((BATCH_SIZE, decode_units))
    y_pred = model_valid([img_tensor, decode_input, hidden])
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.transpose(y_pred, (1, 0))
    padding = tf.equal(y_pred, vocab_size - 1)  # vocab_size - 1: index of <PAD>
    mask = 1 - tf.cast(padding, dtype=tf.float32)
    correct = tf.cast(tf.equal(target, y_pred), dtype=tf.float32) * mask
    accuracy = tf.reduce_sum(correct) / tf.reduce_sum(mask)

    return accuracy


if __name__ == '__main__':

    generator_training = Generator(folder_image='train',
                                   folder_label='train.txt',
                                   batch_size=BATCH_SIZE,
                                   image_height=image_height,
                                   image_width=image_width,
                                   max_txt_length=max_txt_length)
    generator_valid = Generator(folder_image='valid',
                                folder_label='valid.txt',
                                batch_size=BATCH_SIZE,
                                image_height=image_height,
                                image_width=image_width,
                                max_txt_length=max_txt_length)



    if args['finetune']:
        model_train.load_weights('model.h5')

    print(len(generator_training.examples))
    print(len(generator_valid.examples))

    step_per_epoch_training = len(generator_training.examples) // BATCH_SIZE
    step_per_epoch_validate = len(generator_valid.examples) // BATCH_SIZE

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        for i in range(step_per_epoch_training):
            imgs, target = next(generator_training.examples_generator())
            total_loss += train_step(imgs, target)

        for i in range(step_per_epoch_validate):
            imgs_valid, target_valid = next(generator_valid.examples_generator())
            accuracy = validate_step(imgs_valid, target_valid)

        if epoch % 10 == 0:
            model_train.save_weights(f'model_epoch{epoch}.h5')

        print('Epoch {}/{} Loss {:.6f} Accuracy {:.6f}'.format(epoch + 1, EPOCHS,
                                                               total_loss / step_per_epoch_training, accuracy))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
