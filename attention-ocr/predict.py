from image_util import ImageUtil
from model import AttentionOCR
from vocabulary import Vocabulary
import os
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

embedding_dim = 256
vocab_size = 43 + 3
max_txt_length = 30 + 3
units = 512
LEARNING_RATE = 0.001
encode_units = 256
decode_units = 512
image_height = 32
image_width = 320
BATCH_SIZE = 1
vocab = Vocabulary()

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
Attention = AttentionOCR(image_width, image_height, vocab_size, encode_units, decode_units, embedding_dim,
                         max_txt_length, False)
model_valid = Attention.build_model_inference()
model_valid.load_weights('model.h5')

image_util = ImageUtil(image_height=32, image_width=320)

img_tensor = image_util.load('test/1.jpg')
img_tensor = tf.expand_dims(img_tensor, 0)

decode_input = tf.expand_dims([vocab.word_index('<START>')] * BATCH_SIZE, axis=1)
hidden = tf.zeros((BATCH_SIZE, decode_units))
y_pred = model_valid([img_tensor, decode_input, hidden])
y_pred = tf.argmax(y_pred, axis=-1)
y_pred = tf.transpose(y_pred, (1, 0))

result = ''
for i in y_pred[0]:
    result += vocab.labels_to_text([i.numpy()]) + ' '

print(result)
