IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

LAMBDA = 100 # Recommended By Authors of the paper

BUFFER_SIZE = 400
BATCH_SIZE = 16

ACTIVATION = 'Mish' # Use '' for Default Activations and 'Mish' otherwise
GENERATOR_ACTIVATION_INDEX = [
    False, False, False, False, False, False, False, False, # Generator Downsampling Blocks
    False, False, False, False, False, False, False # Generator Upsampling Blocks
]
DISCRIMINATOR_ACTIVATION_INDEX = [
    False, False, False, False
]

LEARNING_RATE = 2e-4
EPOCHS = 20
EPOCHS_PER_SAVE = 1

DATASET_TYPE = 'pix2pix' # 'pix2pix', 'colorization', 'segmentation'
SWITCH_TASK = True
EXISTING_DATASETS = {
    'cityscapes' : 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz',
    'edges2handbags' : 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2handbags.tar.gz',
    'edges2shoes' : 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz',
    'facades' : 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
    'maps' : 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz'
}