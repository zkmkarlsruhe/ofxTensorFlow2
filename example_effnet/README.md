# EfficientNet example

This is an example for openFrameworks, which demonstrates how to load and evaluate a ```pretrained``` SavedModel created in python.

### python
Load a EfficientNet model which is trained on imageNet.

```python
model = tf.keras.applications.EfficientNetB0()
```

### openframeworks
Since Cppflow wraps many ```tensorflow ops``` we can use them through the cppflow namespace. Here we load a jpeg picture, cast it to float and add a dimension for the batches.
```c++
auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("my_cat.jpg")));
input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
input = cppflow::expand_dims(input, 0);
```
We can use TensorFlow's arg_max op to receive the highest value in the vector.
```c++
auto maxLabel = cppflow::arg_max(output, 1);
std::cout << "Maximum likelihood: " << maxLabel << std::endl;
```
To access the underlying vector call the template function get_data<T>().
```c++
auto outputVector = output.get_data<float>();
std::cout << "[282] tiger cat: " << outputVector[282]  << std::endl;
```