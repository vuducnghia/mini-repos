# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw/136542#136542
# https://stats.stackexchange.com/questions/95495/guideline-to-select-the-hyperparameters-in-deep-learning
# https://www.phamduytung.com/blog/2019-02-06-choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras/
import csv
import numpy as np
from sklearn.utils import shuffle
from keras.layers import LSTM, Dense, BatchNormalization
from keras.models import Sequential
import matplotlib.pyplot as plt

filename = 'name_gender.csv'
alphabet = 'abcdefghijklmnopqrstuvwxyz'
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
data_X, data_Y = [], []
rows = []
max_length = 0
with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # get name that probability is 1 and length name less 3
        if line_count > 0 and row[2] == '1' and len(row[0]) > 3:
            rows.append(row)
            if len(row[0]) > max_length:
                max_length = len(row[0])

        line_count += 1

for row in rows:
    int_encoded = [char_to_int[char] for char in row[0].lower()]
    onehot_encoded = list()

    for value in int_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    for _ in range(max_length - len(row[0])):
        onehot_encoded.append(np.zeros(len(alphabet)))

    data_X.append(onehot_encoded)
    # print([[0, 1] if row[1] is 'F' else [1, 0]])
    data_Y.append([0, 1] if row[1] is 'F' else [1, 0])

del rows

data_X, data_Y = shuffle(np.array(data_X), np.array(data_Y))
index_train, index_validate = int(0.7 * len(data_X)), int(0.8 * len(data_X))
train_x, validate_x, test_x = data_X[:index_train], data_X[index_train:index_validate], data_X[index_validate:]
train_y, validate_y, test_y = data_Y[:index_train], data_Y[index_train:index_validate], data_Y[index_validate:]

print(f'shape of data train: {train_x.shape}, validate: {validate_x.shape}, test: {test_x.shape}, y: {test_y.shape}')
print(data_Y[0])

def create_history_plot(history, model_name, metrics=None):
    plt.title('Accuracy and Loss (' + model_name + ')')
    if metrics is None:
        metrics = {'accuracy', 'loss'}
    if 'accuracy' in metrics:
        plt.plot(history.history['accuracy'], color='g', label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], color='b', label='Validation Accuracy')
    if 'loss' in metrics:
        plt.plot(history.history['loss'], color='r', label='Train Loss')
        plt.plot(history.history['val_loss'], color='m', label='Validation Loss')
    plt.legend(loc='best')

    plt.tight_layout()


hidden_nodes = 512
model = Sequential()
model.add(LSTM(units=hidden_nodes, return_sequences=False, input_shape=(max_length, len(alphabet)), dropout=0.5))
model.add(BatchNormalization())
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(train_x, train_y, batch_size=100, epochs=50, validation_data=(validate_x, validate_y))

create_history_plot(history, 'model_name', {'accuracy', 'loss'})
plt.savefig('image.jpg')
