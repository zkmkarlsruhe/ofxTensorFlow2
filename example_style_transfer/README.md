# Style Transfer
This example shows realtime style transfer (about 20fps on a GTX 1080).
We highly recommend using a GPU.

### python 
Note: the model uses a function which relies on the size of the tensor. We were not able to evaluate a SavedModel with unkown input tensor dimensions in C++.

This way you can specify the input dimensions.
```python
@tf.function(input_signature=[tf.TensorSpec([None, 512, 512, 3], dtype=tf.float32)])
def model_predict(input_1):
    return {'outputs': network(input_1, training=False)}
```
