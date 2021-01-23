import tensorflow as tf
import numpy as np
import PIL.Image
import cv2
import os


def tensor_to_image(tensor):
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
    

def load_img(path_to_img, max_dim=None, resize=True):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if resize:
        new_shape = tf.cast([256, 256], tf.int32)
        img = tf.image.resize(img, new_shape)

    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        
    img = img[tf.newaxis, :]

    return img


def resolve_video(network, path_to_video, result):
    cap = cv2.VideoCapture(path_to_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(result, fourcc, 30.0, (640,640))

    while cap.isOpened():
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_LINEAR) 

        print('Transfering video...')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = tf.cast(frame[tf.newaxis, ...], tf.float32) / 255.0

        prediction = network(frame)

        prediction = clip_0_1(prediction)
        prediction = np.array(prediction).astype(np.uint8).squeeze()
        prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)

        out.write(prediction)
        cv2.imshow('prediction', prediction)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyALLWindow()


def create_folder(diirname):
    if not os.path.exists(diirname):
        os.mkdir(diirname)
        print('Directory ', diirname, ' createrd')
    else:
        print('Directory ', diirname, ' already exists')       


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)