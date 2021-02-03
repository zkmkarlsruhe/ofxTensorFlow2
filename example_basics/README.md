# Basic example

This is an example for openFrameworks which demonstrates how to load and evaluate a _SavedModel_ created using TensorFlow2.

### TensorFlow2
Create a meaningless computational graph using the new high level API for TensorFlow2 called Keras. 

***Note***: you can either choose the functional model or the simpler sequential model.
```python
import tensorflow as tf

input = tf.keras.Input(shape=(None, None, 3))
output = tf.keras.layers.Conv2D(1, (2,2), padding='same')(input)
model = tf.keras.Model(inputs=input, outputs=output)
```
Compile and train (fit) the graph on random data.
```python
model.compile(optimizer="adam", loss="mean_squared_error")

test_input = np.random.random((128, 32, 32, 3))
test_target = np.random.random((128, 32, 32, 1))
model.fit(test_input, test_target)
```
After training is done export the graph as _SavedModel_ to the examples "bin/data" folder. This way we can easily find it later on.
```python
model.save('../bin/data/model')
```

***Note***: we solely defined the last dimension of the input. The first channel is the number of batches which is never named and always variable (None). Hence the graph has an input structure of (NONE, NONE, NONE, 3).

### openFrameworks
This addon builds upon cppflow. Cppflow wraps the TensorFlow C library and adds a tensor and model class.

#### cppflow:: ops, tensor & model

Call to TensorFlow's fill function which returns a tensor of arbitrary shape.
```C++
cppflow::tensor input = cppflow::fill({10, 9, 17, 3}, 1.0f);
```
Load the model created in Python
```C++
cppflow::model model("model");
```
Infer the model and retrieve the output
```C++
cppflow::tensor output = model(input);
```


#### ofxTF2:: namespace
The `ofxTF2` namespace defines some models and utility functions that simplfy the integration with openFrameworks. 

##### Model
We define a base model class `ofxTF2::Model` that wraps around `cppflow::model` class and mainly allows to load and infer a model relative to _bin/data_. It expects and returns `cppflow::tensor`.
```C++
ofxTF2::Model ofModel("model");
output = ofModel.runModel(input);
```
Later we will take a look at the advanced `ofxTF2::ThreadedModel`.


##### Conversions
Furthermore, you can convert a `cppflow::tensor` to `std::vector`, `ofPixels` or `ofImage` and backwards using the conversion function defined in _ofxTensorFlow2/src/ofxTensorFlow2Utils.h_.
```C++
std::vector<float> outputVector;
ofxTF2::tensorToVector<float>(output, outputVector);
auto backToTensor = ofxTF2::vectorToTensor<float>(outputVector);
```

***Note***: for now, `tensorToVector<T>()`,`tensorToPixels<T>()`and `tensorToImage<T>()` require you to name the correct type `T` that the neural network outputs, i,e it does not convert values. For example, using an int instead of float will not result in trimmed numbers. Use `cppflow::cast` to convert the values of a tensor. 
