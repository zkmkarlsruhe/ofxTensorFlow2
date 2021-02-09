#!/usr/bin/env python
# coding: utf-8

# In[1]:
from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
local_device_protos


# In[2]:


import tensorflow as tf
print('Eager Execution On ->', tf.executing_eagerly())
print('Tensorflow Version ->', tf.__version__)


# In[3]:

from config import *
from src.utils import *
from src.dataset import *
from src.models import *
from src.losses import *
from src.training import *
from tensorflow.keras.utils import plot_model


# In[4]:

path = download_existing_dataset(EXISTING_DATASETS[DATASET], DATASET)
print('Dataset Path ->', path)


# In[5]:

# visualize(join(path, 'train/200.jpg'))


# In[6]:

# visualize(join(path, 'train/200.jpg'), augment=True)


# In[7]:

train_dataset, test_dataset = get_datasets(path)
print(train_dataset)
print(test_dataset)


# In[8]:

generator = Generator()
generator.summary()
# plot_model(
#     generator, rankdir='TB',
#     to_file='generator.png',
#     show_shapes=True,
#     show_layer_names=True,
# )


# In[9]:


discriminator = Discriminator()
discriminator.summary()
# plot_model(
#     discriminator, rankdir='TB',
#     to_file='discriminator.png',
#     show_shapes=True,
#     show_layer_names=True,
# )


# In[10]:


discriminator_optimizer, generator_optimizer = get_optimizers()
checkpoint, checkpoint_prefix = get_checkpoint(
    discriminator, generator,
    discriminator_optimizer,
    generator_optimizer
)


# In[ ]:


generator_loss_history, discriminator_loss_history = train(
    discriminator, generator,
    discriminator_optimizer,
    generator_optimizer,
    train_dataset, test_dataset,
    checkpoint, checkpoint_prefix
)


# In[ ]:


for _input, _target in test_dataset.take(5):
    generate_images(generator, _input, _target)
