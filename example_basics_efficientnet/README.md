# EfficientNet example

This example demonstrates how to load and evaluate a _pretrained_ graph.

### TensorFlow2
Load an EfficientNet model which is pretrained on the ImageNet dataset consisting of 1000 classes to solve the image classification task.

***Note***: EfficientNet is an upscalable architecture. TensorFlow has 8 different version. Try different models by changing the last number (i.e. EfficientNetB7) and observe the increase in parameters, computation time and accuracy.

```python
model = tf.keras.applications.EfficientNetB0()
```

### openFrameworks
Since cppflow wraps many TensorFlow operations (ops) we can use them through the cppflow namespace. Here we load a jpeg picture, cast it to float and add a dimension for the batches.
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
To convert the ouput tensor to a vector call the function `ofxTF2::tensorToVector<T>()`.
```c++
std::vector<float> outputVector;
ofxTF2::tensorToVector(output, outputVector);
std::cout << "[282] tiger cat: " << outputVector[282]  << std::endl;
```
