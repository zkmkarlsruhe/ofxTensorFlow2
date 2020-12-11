from config import *
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


loss = BinaryCrossentropy(from_logits = True)


def discriminator_loss(disc_real_output, disc_generated_output):
    '''Pix2Pix Discriminator Loss
    Reference: https://arxiv.org/abs/1611.07004
    Params:
        disc_real_output        -> Real Image passed to Discriminator
        disc_generated_output   -> Generated Image passed to Discriminator
    '''
    real_loss = loss(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    '''Pix2Pix Discriminator Loss
    Reference: https://arxiv.org/abs/1611.07004
    Params:
        disc_generated_output   -> Generated Image passed to Discriminator
        gen_output              -> Generator Output
        target                  -> Ground Truth
    '''
    gan_loss = loss(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss