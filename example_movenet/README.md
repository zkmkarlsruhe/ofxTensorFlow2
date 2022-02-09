# Movenet example
This is an example for realtime 2D multiuser skeleton tracking over an RGB camera or video using the MoveNet model.

Video sample from Polina Tankilevitch  
https://www.pexels.com/video/video-of-women-dancing-3873059/


![](../media/movenet.gif)


Example made with love by Natxopedreira 2021  
https://github.com/natxopedreira


### TensorFlow2 
For this example we do not have the python code that produced the model. However, a SavedModel of MoveNet has been uploaded to [TensorFlow Hub](https://tfhub.dev).

### openFrameworks
As with other examples, we will make use of the `ofxTF2::ThreadedModel`. Please take a look at other examples for more information on how to use the class.

Taking a look at the output of the `saved_model_cli` tool we find that this MoveNet model expects the input to be a color image of dimensions [1, height, width, 3].
```shell
dtype: DT_INT32
shape: (1, -1, -1, 3)
```
__NOTE__: Remember the first dimension is always the batch size which is usually 1 in realtime applications. The second and third dimensions need to be a multiple of 32 and the larger dimension is recommended to be 256.

And outputs a vector of 6 skeletons.
```shell
dtype: DT_FLOAT
shape: (1, 6, 56)
```
Each skeleton contains 56 values with defined meaning. The first 51 values make up the parameters of the 17 joints or key points.
Each key point consists of three values: x and y position and the confidence of the network it has for that key point.

The other 5 values are the bounding box of that skeleton and again the networks confidence.


### Further Reading
https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html