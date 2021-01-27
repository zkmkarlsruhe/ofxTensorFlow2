from modules.utils import tensor_to_image, load_img, clip_0_1, resolve_video
from modules.forward import feed_forward
import tensorflow as tf
import os
import argparse

def convert(src_dir, dest_dir, shape):

    # Build the feed-forward network and load the weights.
    network = feed_forward()
    network.load_weights(os.path.join(src_dir,"weights")).expect_partial()

    model_name = os.path.join(dest_dir, src_dir.split('/')[-1])

    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    network.build(tuple(shape))

    @tf.function(input_signature=[tf.TensorSpec(shape, dtype=tf.float32)])
    def model_predict(input_1):
        return {'outputs': network(input_1, training=False)}

    network.save(model_name, signatures={'serving_default': model_predict})


if __name__ == "__main__":

    #parser = argparse.ArgumentParser(
    #    description='Fast Style Transfer')
    #parser.add_argument('--path', required=True, type=str)

    path = os.path.dirname('../bin/data/chkpt_models/')
    dest = os.path.dirname('../bin/data/models/')

    shape = [None, 480, 640, 3]

    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    for subpath in subfolders:
        print("Converting", shape, ": ", subpath, " to ", dest)
        convert(subpath, dest, shape)

