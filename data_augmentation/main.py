from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

img = load_img('data/train/cats/cat.0.jpg')
x = img_to_array(img)
print(x.shape)
x = x.reshape((1,) + x.shape)
# print((1,) + x.shape)
i = 1
for batch in datagen.flow(x = x, batch_size=2, save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break