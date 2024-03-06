# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# https://medium.com/@vtunanh_15103/t%C3%ACm-m%E1%BB%91i-li%C3%AAn-h%E1%BB%87-gi%E1%BB%AFa-c%C3%A1c-tag-v%E1%BB%9Bi-nhau-b%E1%BA%B1ng-word-embedding-fa15aa2ff3f5
# https://machinelearningmastery.com/what-are-word-embeddings/
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding

docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define class labels
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# estimate the vocabulary size of 50, which is much larger than needed to reduce the probability of collisions from the hash function.
vocab_size = 50
encoded_docs = [one_hot(text=d, n=vocab_size) for d in docs]
print(encoded_docs)

max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs, padded_docs.shape)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
model.summary()

model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
