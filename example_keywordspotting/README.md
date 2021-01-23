# Keywordspotting

This example uses code from an external repository to spot keywords in audio signals. Fortunately, the developers deliver an .h5 file, which contains the weights of a trained neural network. In Python we reconstruct the model, load the weights and export the model as SavedModel.

Check out the python notebook train.py if you want to learn more about the training process.

### TensorFlow2
After loading the model we use a trick to save the model. It allows us to change the signature of the SavedModel and to specify `training=false`. The later is very important as some layers (such as Dropout) act different during training and will not get correctly initialized in C++. 

```python
@tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.float32)])
def model_predict(input_1):
  return {'outputs': model(input_1, training=False)}

model.save('../bin/data/model', signatures={'serving_default': model_predict})
```

### openFrameworks
We will use the microphone so be sure to have the right settings. 

Since the neural network was trained on 1 seconds long audio files sampled at 16kHz we will need to assure the same effective sampling rate and cut the audio stream accordingly.

We use a sampling rate of 48kHz for the microphone as it is easily convertable to 16kHz. To collect 1s long audio snippets we start recording after a certain volume treshhold is surpassed. As this introduces some latency we keep the previous audio buffer and add it to the 1s long snippet as soon as we start recording.

Try to adjust the sampling rate if your microphone is suited, but remember to adjust the downsampling factor.

***Note***: that all preprocessing is done inside the computational graph. Wow! Thanks _kapre_.
