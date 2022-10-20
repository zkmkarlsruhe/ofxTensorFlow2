# Realtime Neural Style Transfer
This is an example for realtime (\~30fps @ RTX 2070 Super) neural style transfer in openFrameworks.

![GIF Style Transfer](../media/style_transfer.gif)

The Python code has been taken from [this repo](https://github.com/cryu854/FastStyle). Neural style transfer can also be done using GANs. However, this method is much faster. Check this [post](https://www.tensorflow.org/tutorials/generative/style_transfer?hl=en) for more information on this topic.
Nevertheless, we highly recommend using a GPU.

Make sure to download and extract the folder containing multiple _SavedModels_ from the assets. The easiest way is to use the script provided in _../scripts/download_example_model.sh_. The application will by default look for a folder _models_ in _bin/data/_.

### TensorFlow2 
You can download the checkpoints for each model using the bash script _../scripts/download_training_examples.sh_.

This example inherits a small problem. The computational graph uses a function which relies on the size of the tensor. This leads to an error when saving the model. We have to specify the input dimensions using the following wrapper:
```python
@tf.function(input_signature=[tf.TensorSpec([1, 480, 640, 3], dtype=tf.float32)])
def model_predict(input_1):
    return {'outputs': network(input_1, training=False)}
```
***Note***: Run the python script _python/checkpoint2SavedModel.py_ on the downloadable checkpoints to change the input signatures!

### openFrameworks
***For GPU users***: by default TensorFlow will try to reserve almost all GPU memory, independent of the model size. We define certain presets to change this behaviour. You can choose between 10 to 90 % reservation and with or without memory growth.
```c++
// restrict TensorFlow to reserve only a maximum of 70% of the GPUs memory
// and set memory growth to true
ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true);
```

In this example we will use the `ThreadedModel` class and augment the runModel function. This way we can modify the in and outputs inside the thread. 

Here, the model expects a 3D tensor (which has no dimension for batches yet) and outputs the values to be displayable without clamping (the neural network applies a weird shift).
```c++
class ImageToImageModel : public ofxTF2::ThreadedModel {
    public:
    // override the runModel function of ofxTF2::ThreadedModel
    // this way the thread will take this augmented function 
    // otherwise it would call runModel with no way of pre/post-processing
    cppflow::tensor runModel(const cppflow::tensor & input) const override {
        // cast data type and expand to batch size of 1
        auto inputCast = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        inputCast = cppflow::expand_dims(inputCast, 0);
        // call to super 
        auto output = ofxTF2Model::runModel(inputCast);
        // postprocess: last layer = (tf.nn.tanh(x) * 150 + 255. / 2)
        return ofxTF2::mapTensorValues(output, -22.5f, 277.5f, 0.0f, 255.0f);
    }
};
```
***Note***: The default layout for images in TensorFlow is NHWC (batch size, height, width, channel), which is different to openFrameworks' layout (width, height, channel). So an image which is 640 pixels wide and 480 pixels high is given as an tensor of `[1, 480, 640, 3]`.
Check this [link](https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html) for more details.
