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

# load the efficientnet model
model = tf.keras.applications.EfficientNetB0()

# export to a SavedModel
model.save('../bin/data/model')
