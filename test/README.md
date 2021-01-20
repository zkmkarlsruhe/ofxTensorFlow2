# Basic example

This is an example for openFrameworks, which demonstrates how to load and evaluate a SavedModel created in python.

### python
create a computational graph using tensorflow2. Note: besides the functional model there is also the simpler sequential model.
```python
import tensorflow as tf

input = tf.keras.Input(shape=(None, None, 3))
output = tf.keras.layers.Conv2D(1, (2,2), padding='same')(input)
model = tf.keras.Model(inputs=input, outputs=output)
```
export the SavedModel always to ../bin/data. This way we can easily find it later on.
```
model.save('../bin/data/model')
```

### openframeworks

allocate and fill a tensor of arbitrary shape in c++. Note: in python we solely named the dimension of the last channel (here: 3). The first channel is the number of batches, not namened and always variable (None)

```c++
auto input = cppflow::fill({10, 9, 17, 3}, 1.0f);
```
load the model created in python
```c++
cppflow::model model(ofToDataPath("model"));
```
infer the model and retrieve the output
```c++
auto output = model(input);
std::cout << output << std::endl;
```