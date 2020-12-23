import tensorflow as tf

# load the efficientnet model
model = tf.keras.applications.EfficientNetB0()

# export to a SavedModel
model.save('model', save_format='tf')


