# https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/
import os
from imutils import paths
import shutil
import random
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import classification_report

ORIG_INPUT_DATASET = "Food-5K"
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"
BASE_CSV_PATH = "output"
BASE_PATH = "dataset"
LE_PATH = os.path.sep.join(["output", "le.cpickle"])

CLASSES = ['non-food', 'food']
BATCH_SIZE = 32


def build_dataset():
    for split in (TRAIN, TEST, VAL):
        print(f'[INFO] processing {split} split')
        p = os.path.sep.join([ORIG_INPUT_DATASET, split])
        imagePaths = list(paths.list_images(p))

        for imagePath in imagePaths:
            filename = imagePath.split(os.path.sep)[-1]
            label = CLASSES[int(filename.split("_")[0])]

            dirPath = os.path.sep.join([BASE_PATH, split, label])

            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            p = os.path.sep.join([dirPath, filename])
            shutil.copy2(imagePath, p)


def extract_feature():
    print("[INFO] loading network...")
    model = ResNet50(weights="imagenet", include_top=False)
    le = None

    for split in (TRAIN, TEST, VAL):
        # grab all image paths in the current split
        print("[INFO] processing '{} split'...".format(split))
        p = os.path.sep.join([BASE_PATH, split])
        imagePaths = list(paths.list_images(p))

        # randomly shuffle the image paths and then extract the class
        # labels from the file paths
        random.shuffle(imagePaths)
        labels = [p.split(os.path.sep)[-2] for p in imagePaths]

        # if the label encoder is None, create it
        if le is None:
            le = LabelEncoder()
            le.fit(labels)

        # open the output CSV file for writing
        csvPath = os.path.sep.join([BASE_CSV_PATH, "{}.csv".format(split)])
        csv = open(csvPath, "w")

        for (b, i) in enumerate(range(0, len(imagePaths), BATCH_SIZE)):
            print(f'[INFO] processing batch {b + 1}/{int(np.ceil(len(imagePaths) / float(BATCH_SIZE)))}')

            batchPaths = imagePaths[i:i + BATCH_SIZE]
            batchLabels = le.transform(labels[i:i + BATCH_SIZE])
            batchImages = []

            for imagePath in batchPaths:
                image = load_img(imagePath, target_size=(224, 224))
                image = img_to_array(image)

                image = np.expand_dims(image, axis=0)
                # subtracting the mean RGB pixel intensity from the ImageNet dataset
                image = imagenet_utils.preprocess_input(image)

                batchImages.append(image)

            batchImages = np.vstack(batchImages)
            features = model.predict(batchImages, batch_size=BATCH_SIZE)
            features = features.reshape((features.shape[0], 7 * 7 * 2048))

            for (label, vec) in zip(batchLabels, features):
                vec = ",".join([str(v) for v in vec])
                csv.write("{},{}\n".format(label, vec))
        csv.close()
    f = open(LE_PATH, "wb")
    f.write(pickle.dumps(le))
    f.close()


def csv_feature_generator(inputPath, bs, numClasses, mode='train'):
    f = open(inputPath, 'r')
    while True:
        data, labels = [], []

        while len(data) < bs:
            row = f.readline()

            if row == '':
                # reset the file pointer to the beginning of the file and re-read the row
                f.seek(0)
                row = f.readline()

                if mode == "eval":
                    break

            # extract the class label and features from the row
            row = row.strip().split(",")
            label = row[0]
            label = to_categorical(label, num_classes=numClasses)
            features = np.array(row[1:], dtype="float")

            data.append(features)
            labels.append(label)
        yield np.array(data), np.array(labels)


if __name__ == '__main__':
    # build_dataset()
    # extract_feature()

    le = pickle.loads(open(LE_PATH, "rb").read())

    # derive the paths to the training, validation, and testing CSV files
    trainPath = os.path.sep.join([BASE_CSV_PATH, "{}.csv".format(TRAIN)])
    valPath = os.path.sep.join([BASE_CSV_PATH, "{}.csv".format(VAL)])
    testPath = os.path.sep.join([BASE_CSV_PATH, "{}.csv".format(TEST)])

    totalTrain = sum([1 for l in open(trainPath)])
    totalVal = sum([1 for l in open(valPath)])

    testLabels = [int(row.split(",")[0]) for row in open(testPath)]
    totalTest = len(testLabels)

    trainGen = csv_feature_generator(trainPath, BATCH_SIZE, len(CLASSES), mode="train")
    valGen = csv_feature_generator(valPath, BATCH_SIZE, len(CLASSES), mode="eval")
    testGen = csv_feature_generator(testPath, BATCH_SIZE, len(CLASSES), mode="eval")

    model = Sequential()
    model.add(Dense(256, input_shape=(7 * 7 * 2048,), activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(len(CLASSES), activation="softmax"))
    opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3 / 25)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] training simple network...")

    H = model.fit_generator(
        trainGen,
        steps_per_epoch=totalTrain // BATCH_SIZE,
        validation_data=valGen,
        validation_steps=totalVal // BATCH_SIZE,
        epochs=25)

    print("[INFO] evaluating network...")
    predIdxs = model.predict_generator(testGen,
                                       steps=(totalTest // BATCH_SIZE) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testLabels, predIdxs,
                                target_names=le.classes_))