import tensorflow as tf
from config import *
from os.path import join
from matplotlib import pyplot as plt


def load(image_file):
    '''Load an image from a file
    Params:
        image_file  -> Path to an image file
    '''
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image.set_shape([None, None, 3])
    if DATASET_TYPE == 'pix2pix':
        w = tf.shape(image)[1]
        w = w // 2
        real_image = image[:, : w, :]
        input_image = image[:, w :, :]
    elif DATASET_TYPE == 'colorization':
        gray_image = tf.image.rgb_to_grayscale(image)
        input_image = tf.concat([gray_image, gray_image, gray_image], axis =-1)
        real_image = image
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    if SWITCH_TASK == True:
        return real_image, input_image
    else:
        return input_image, real_image


def visualize(image_file, augment = False):
    '''Visualization of a single data point
    Params:
        image_file  -> Path to an image file
        augment     -> Apply augmentation (Flag)
    '''
    input_image, real_image = load(image_file)
    if augment:
        input_image, real_image = augmentation(input_image, real_image)
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 16))
    plt.setp(axes.flat, xticks = [], yticks = [])
    # for i, ax in enumerate(axes.flat):
    #     if i % 2 == 0:
    #         ax.imshow(input_image.numpy() / 255.0)
    #         ax.set_xlabel('Input_Image')
    #     else:
    #         ax.imshow(real_image.numpy() / 255.0)
    #         ax.set_xlabel('Real_Image')
    # plt.show()


def resize(input_image, real_image, height, width):
    '''Resize
    Params:
        input_image -> Input Image
        real_image  -> Real Image (ground truth)
        height      -> Height of the image
        width       -> Width of the image
    '''
    input_image = tf.image.resize(
        input_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image, real_image


def random_crop(input_image, real_image):
    '''Random Crop
    Params:
        input_image -> Input Image
        real_image  -> Real Image (ground truth)
    '''
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image,
        size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    '''Normalize Images
    Params:
        input_image -> Input Image
        real_image  -> Real Image (ground truth)
    '''
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def augmentation(input_image, real_image):
    '''Apply random augmentation
    Params:
        input_image -> Input Image
        real_image  -> Real Image (ground truth)
    '''
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image


def load_image_train(image_file):
    '''Load Training Images
    Params:
        image_file  -> Path to an image file
    '''
    input_image, real_image = load(image_file)
    input_image, real_image = augmentation(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_image_test(image_file):
    '''Load Test Images
    Params:
        image_file  -> Path to an image file
    '''
    input_image, real_image = load(image_file)
    input_image, real_image = resize(
        input_image, real_image,
        IMG_HEIGHT, IMG_WIDTH
    )
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def get_datasets(path):
    '''Get Training and testing datasets
    Params:
        path    -> Dataset Path
    '''
    train_dataset = tf.data.Dataset.list_files(join(path, 'train/*.jpg'))
    train_dataset = train_dataset.map(
        load_image_train,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.list_files(join(path, 'val/*.jpg'))
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset, test_dataset
