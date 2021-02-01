# Keywordspotting
This example uses code from an [this repo](https://github.com/douglas125/SpeechCmdRecognition) to spot keywords in audio signals. 

We will use the microphone so be sure to have the right setting.

The following keywords can be recognized: ["Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree",  "Wow"]

### TensorFlow2
Fortunately, the authors deliver an .h5 file, which contains the weights of a trained neural network. In Python we reconstruct the model, load the weights and export the model as _SavedModel_.
After loading the .h5 file we wrap the call to the model. This allows us to change the signature of the _SavedModel_ and to specify `training=False`. The later is very important as some layers (such as Dropout) act different during training and will not get correctly initialized in C++. 

```python
@tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.float32)])
def model_predict(input_1):
  return {'outputs': model(input_1, training=False)}

model.save('../bin/data/model', signatures={'serving_default': model_predict})
```

***Note***: besides downsampling all preprocessing is done inside the computational graph. Wow! Thanks to the pyhton package _kapre_.

***Note***: check out train.py if you want to learn more about the training process.


### openFrameworks
Since the neural network was trained on 1 seconds long audio files sampled at 16kHz we will need to assure the same effective sampling rate and cut the audio stream accordingly.
We use a sampling rate of 48kHz for the microphone as it is easily convertable to 16kHz. To collect 1s long audio snippets we start recording after a certain volume threshold is surpassed. As this introduces some latency we keep the previous audio buffers in a FiFo and add them as soon as we start recording.

Try to adjust the sampling rate and/or audio buffer size to suit the needs of your microphone, but remember to adjust the downsampling factor. For now, downsampling is only support by integers of 16kHz.

In this example you will find a specification of ofxTF2Model that adds a classification and downsampling method.
The AudioClassifier excepts a FiFo of audio buffers, applies downsampling, infers the neural network and returns the element with the highest probability.
```C++
class AudioClassifier : public ofxTF2Model {
	public:
	void classify(AudioBufferFifo & bufferFifo, int downsamplingFactor,
			int & argMax, float & prob);
	private: 
	void downsample(AudioBufferFifo & bufferFifo, int downsamplingFactor);

	SimpleAudioBuffer sample_;
};
```