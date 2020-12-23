import tensorflow as tf
import numpy as np

# create a model
# the input dimensions of e.g. an image can be unknown
# though for the Conv2D layer the channels must be known 
input = tf.keras.Input(shape=(None, None, 3))
output = tf.keras.layers.Conv2D(1, (2,2), padding='same')(input)
model = tf.keras.Model(inputs=input, outputs=output)

model.compile(optimizer="adam", loss="mean_squared_error")

# train the model
test_input = np.random.random((128, 32, 32, 3))
test_target = np.random.random((128, 32, 32, 1))
model.fit(test_input, test_target)

# export to a SavedModel
model.save('model')

# test: load and predict
reconstructed = tf.keras.models.load_model('model')
out = reconstructed.predict(test_input)
print(out)