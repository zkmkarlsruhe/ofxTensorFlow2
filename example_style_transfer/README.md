# Realtime Neural Style Transfer
This is an example for realtime style transfer (about 20fps on a GTX 1080) in openFrameworks. 

Neural style transfer can also be done using GANs. However, this method is much faster. Check this [post](https://www.tensorflow.org/tutorials/generative/style_transfer?hl=en) for more information on this topic.
Nevertheless, we highly recommend using a GPU.

Make sure to download and extract the folder containing multiple _SavedModels_ from the assets. The application will by default look for a folder "models/" in "example_style_transfer/bin/data/".

### TensorFlow2 
***Note***: the computational graph uses a function which relies on the size of the tensor. This leads to a problem when saving the model. We have to specify the input dimensions using the already presented wrapper, e.g:
```python
@tf.function(input_signature=[tf.TensorSpec([None, 480, 640, 3], dtype=tf.float32)])
def model_predict(input_1):
    return {'outputs': network(input_1, training=False)}
```
***Note***: you can download the checkpoints for each model from our assets and use the script "checkpoint2SavedModel.py" to change the input signature!

### openFrameworks

***Note***: The default layout for images in TensorFlow is NHWC (batch, height, width, channel), which is different to openframeWorks' layout (width, height, channel). Check this [link](https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html) for more details.