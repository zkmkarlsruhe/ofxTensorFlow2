# Frozen Graph Legacy Support - CharRnn
This example demonstrates how to load and run a frozen graph developed in TensorFlow. More specific, we demonstrate how to port an example of _ofxMSATensorFlow_ to _ofxTensorFlow2_.

![](../media/charRnn.gif)

### TensorFlow
From Memo Akten's example:

> Models are trained and saved in python with this code (https://github.com/memo/char-rnn-tensorflow)
and loaded in openframeworks for prediction.
I'm supplying a bunch of pretrained models (bible, cooking, erotic, linux, love songs, shakespeare, trump),
and while the text is being generated character by character (at 60fps!) you can switch models in realtime mid-sentence or mid-word.
(Drop more trained models into the folder and they'll be loaded too).
Note, all models are trained really quickly with no hyperparameter search or cross validation,
using default architecture of 2 layer LSTM of size 128 with no dropout or any other regularisation.

### openFrameworks
By default `ofxTF2::Model` will use the `SavedModel` format. To use the `FrozenGraph` format you may either add the type to the constructor or call `Model::setModelType()` afterwards.
As the default names differ from the names in the `SavedModel` format make sure to overwrite names of the ins and outs by calling `Model::setup()`.
```c++
// set model type and i/o names
model.setModelType(cppflow::model::TYPE::FROZEN_GRAPH);
std::vector<std::string> inputNames = {
	"data_in",
	"state_in",
};
std::vector<std::string> outputNames = {
	"data_out",
	"state_out",
};
model.setup(inputNames, outputNames);
```

Afterwards you can load the _pb file_ using the `Model::load()` function.
```c++
if(!model.load(model_path)) {
	std::exit(EXIT_FAILURE);
}
```
This approach is different to _ofxMSATensorFlow_ where you would name the input and output tensors when running the model.
Lets have a look at the way we ran the model in _ofxMSATensorFlow_:
```c++
// run the model (ofxMSATensorFlow)
std::vector<tensorflow::Tensor> t_out;
std::vector<std::string> fetch_tensors = { "data_out", "state_out" };
session->Run({ { "data_in", t_data_in }, { "state_in", t_state_in } }, fetch_tensors, {}, &t_out);
```
In _ofxTensorFlow2_ we can use the `Model::run()` function (if we have a single input and output tensor) or the more general `Model::runMultiModel()` function to execute the model on some input data:
```c++
// run the model (ofxTensorFlow2)
std::vector<cppflow::tensor> vectorOfInputTensors = {t_data_in, t_state_in};
auto vectorOfOutputTensors = model.runMultiModel(vectorOfInputTensors);
```

The class for the tensor also differs a little bit. In _ofxTensorFlow2_ we use the `cppflow::tensor` class instead of `tensorflow::tensor`.

Similar to _ofxMSATensorFlow_, we provide functions for converting from a tensor to `ofImage, ofPixels` and `std::vector`, and the other way around, e.g. `ofxTF2::vectorToTensor()`. Regarding other utilities, there are now some new handy functionalities like `ofxTF2::setGPUMaxMemory()` and `ofxTF2::ThreadedModel`, while other less general functions have been dropped. 
For this example we have put dropped functions into the source code of the example.

__Note:__ With the current version of _cppflow_, calling `tensor::get_data()` or `tensor::shape()` on a default (uninitialized) `cppflow::tensor` will result in a segfault. Wile you usually don't need to call them yourself, this effects the previously mentioned conversion functions.
Therefore, we've changed a few lines of code when priming the model with new data.

Please understand that we wont be able to invest a lot of time in supporting this feature in the future.