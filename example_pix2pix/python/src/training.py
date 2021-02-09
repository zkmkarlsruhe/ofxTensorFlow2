import time
from config import *
from .losses import *
from os.path import join
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model


def get_optimizers():
    '''Build optimizers for Generator and Discriminator'''
    generator_optimizer = Adam(LEARNING_RATE, beta_1=0.5)
    discriminator_optimizer = Adam(LEARNING_RATE, beta_1=0.5)
    return discriminator_optimizer, generator_optimizer


def get_checkpoint(
    discriminator, generator,
    discriminator_optimizer,
    generator_optimizer,
    checkpoint_dir='./training_checkpoints'):

    '''Get Training Checkpoint
    Params:
        discriminator           -> Discriminator
        generator               -> Generator
        discriminator_optimizer -> Discriminator Optimizer
        generator_optimizer     -> Generator Optimizer
        checkpoint_dir          -> Checkpoint Directory
    '''

    checkpoint_prefix = join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator, discriminator=discriminator
    )
    return checkpoint, checkpoint_prefix


def generate_images(model, test_input, tar):
    '''Generate Images
    Params:
        model       -> Generator Model
        test_input  -> Input Image for testing
        tar         -> Target Ground Truth
    '''
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def train(
    discriminator, generator,
    discriminator_optimizer,
    generator_optimizer,
    train_dataset, test_dataset,
    checkpoint, checkpoint_prefix,
    checkpoint_step = EPOCHS_PER_SAVE,
    save_checkpoints = False):

    '''Training Function
    Params:
        discriminator           -> Discriminator
        generator               -> Generator
        discriminator_optimizer -> Discriminator Optimizer
        generator_optimizer     -> Generator Optimizer
        train_dataset           -> Train Dataset
        test_dataset            -> Test Dataset
        checkpoint              -> Checkpoint Object
        checkpoint_prefix       -> Checkpoint Prefix
    '''

    generator_loss_history, discriminator_loss_history = [], []

    @tf.function
    def train_step(input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            generator_loss_history.append(gen_loss)
            discriminator_loss_history.append(disc_loss)

        generator_gradients = gen_tape.gradient(
            gen_loss,
            generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss,
            discriminator.trainable_variables
        )
        generator_optimizer.apply_gradients(
            zip(
                generator_gradients,
                generator.trainable_variables
            )
        )
        discriminator_optimizer.apply_gradients(
            zip(
                discriminator_gradients,
                discriminator.trainable_variables
            )
        )


    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], dtype=tf.float32)])
    def model_predict(input_1):
        return {'outputs': generator(input_1, training=False)}

    def fit(train_ds, test_ds, epochs):
        for epoch in range(epochs):
            start = time.time()
            # Train
            print('Epoch', str(epoch + 1), 'going on....')
            for input_image, target in tqdm(train_ds):
                train_step(input_image, target)
            print('Completed.')

            # saving the model
            if (epoch + 1) % checkpoint_step == 0:
                generator.save("../bin/data/model"+str(epoch), signatures={'serving_default': model_predict})
                if save_checkpoints:
                    checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


    fit(train_dataset, test_dataset, EPOCHS)

    return generator_loss_history, discriminator_loss_history
