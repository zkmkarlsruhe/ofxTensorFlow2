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

### openFrameworks

***Note***: when we created the graph in Python we solely definded the last dimension of the input. The first channel is the number of batches which is never named and always variable (None). Hence the graph has an input structure of (NONE, NONE, NONE, 3).

Allocate and fill a tensor of arbitrary shape in C++.
```c++
auto input = cppflow::fill({10, 9, 17, 3}, 1.0f);
```
Load the model created in python
```c++
cppflow::model model(ofToDataPath("model"));
```
Infer the model and retrieve the output
```c++
auto output = model(input);
std::cout << output << std::endl;
```
