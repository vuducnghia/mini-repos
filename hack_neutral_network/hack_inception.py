# https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b
# https://stackoverflow.com/questions/50098971/whats-the-difference-between-gradienttape-implicit-gradients-gradients-functi
# https://stackoverflow.com/questions/58322147/how-to-generate-cnn-heatmaps-using-built-in-keras-in-tf2-0-tf-keras
# https://gist.github.com/haimat/10a53ad9675f8f5ac1290f06c3e4f973
# hack inception predict dog to water_buffalo
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import inception_v3
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def predict(model):
    prediction = model.predict(origin_image)
    predicted_classes = inception_v3.decode_predictions(prediction, top=1)
    imagenet_id, name, confidence = predicted_classes[0][0]
    print(f'name: {name}, confident: {confidence}')


def grad(cost_function, model_input_layer):
    with tf.GradientTape() as t:
        grad = t.gradient(cost_function, model_input_layer)
        # print(grad)
    return grad


if __name__ == '__main__':
    # load pre-trained image recognition
    model = inception_v3.InceptionV3()

    origin_image = load_img("dog.jpg", target_size=(299, 299))
    origin_image = img_to_array(origin_image)
    origin_image = (origin_image - 127.5) / 127.5
    origin_image = np.expand_dims(origin_image, axis=0)

    hacked_image = np.copy(origin_image)
    hacked_image = tf.convert_to_tensor(hacked_image)
    print(f'hacked_image: {hacked_image.shape}')

    confident = 0.0
    learning_rate = 0.1
    id_water_buffalo = 346
    model.summary()

    conv_layer = model.get_layer("input_1")
    heatmap_model = Model([model.inputs], [conv_layer.input, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(hacked_image)
        print(f'inputs: {heatmap_model.inputs}')
        loss = predictions[:, np.argmax(predictions[0])]
        # print(f'loss: {loss}')
        print(f'conv_output: {conv_output.shape}')
        grads = gtape.gradient(loss, conv_output)
        print(grads)
    # while confident < 0.8:
    #     out = model(hacked_image)
    #     print(model.inputs)
    #     # model_output_layer = model.layers[-1].output
    #     # print(out.shape)
    #
    #     cost_function = out[0, id_water_buffalo]
    #
    #     with tf.GradientTape() as t:
    #         grad = t.gradient(cost_function, model.inputs)
    #         print(grad)
    #     break