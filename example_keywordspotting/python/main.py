# ofxTensorFlow2
#
# Copyright (c) 2021 ZKM | Hertz-Lab
# Paul Bethge <bethge@zkm.de>
#
# BSD Simplified License.
# For information on usage and redistribution, and for a DISCLAIMER OF ALL
# WARRANTIES, see the file, "LICENSE.txt," in this distribution.
#
# This code has been developed at ZKM | Hertz-Lab as part of „The Intelligent 
# Museum“ generously funded by the German Federal Cultural Foundation.


import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# load the model from weights
model = tf.keras.models.load_model('model-attRNN.h5', custom_objects={
          'Melspectrogram': Melspectrogram,
          'Normalization2D': Normalization2D })
model.summary()


# save model
@tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.float32)])
def model_predict(input_1):
  return {'outputs': model(input_1, training=False)}

model.save('../bin/data/model', signatures={'serving_default': model_predict})


# if you want to test the model set test to True and sample to a path
test = False
if test:
  import scipy.io.wavfile as wav
  sample = 'sample.wav'
  model = tf.keras.models.load_model('../bin/data/model')
  audio = wav.read(sample)
  data_tensor = tf.convert_to_tensor(audio[1])
  data_tensor = tf.expand_dims(data_tensor, axis=0)
  print(data_tensor.shape.as_list())

  out = model(data_tensor)
  print(out)
