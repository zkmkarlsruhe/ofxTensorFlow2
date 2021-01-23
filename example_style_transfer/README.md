# Realtime Neural Style Transfer
This is an example for realtime style transfer (about 20fps on a GTX 1080) in openFrameworks. 

Neural style transfer can also be done using GANs. However, this method is much faster. Check this [post](https://www.tensorflow.org/tutorials/generative/style_transfer?hl=en) for more information on this topic.
Still, we highly recommend using a GPU.

### TensorFlow2 
***Note***: the computational graph uses a function which relies on the size of the tensor. This lead to a problem when loading the model in C++, but not in Python. We were not able to evaluate a _SavedModel_ with unkown input tensor dimensions and therefor had to specify a it using the already presented wrapper.
```python
@tf.function(input_signature=[tf.TensorSpec([None, 640, 480, 3], dtype=tf.float32)])
def model_predict(input_1):
    return {'outputs': network(input_1, training=False)}
```
***Note***: you can load a pretrained model from our assets and change the input signature during export!
