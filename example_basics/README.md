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
Models can also run on multiple input and output tensors. Check _example_basics_multi_io_ for more details. In _example_pix2pix_  and _example_style_transfer_ we will explore the usage of the derived class `ofxTF2::ThreadedModel` for asynchronous processing.


##### Utility
The utility functions defined in _ofxTensorFlow2/src/ofxTensorFlow2Utils.h_ may
come in handy for average usage. A common need is to convert a 
`cppflow::tensor` to `std::vector`, `ofPixels` or `ofImage` or the other way around.
```C++
// cast the TF_FLOAT output tensor to TF_INT, then copy to int vector
std::vector<int> outputVector;
ofxTF2::tensorToVector<int>(output, outputVector);
//  and back to TF_INT tensor
auto backToTensor = ofxTF2::vectorToTensor<int>(outputVector); 
``` 
***Note***: it is not necessary to name the template type. The output type will always depend on the std::vector, ofImage, or ofPixel given. For example, calling` ofxTF2::imageToTensor<T>()` on an ofFloatImage will return a tensor of TF_FLOAT (the TF version of float).