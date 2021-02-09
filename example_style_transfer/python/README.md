# Fast Style Transfer
The code has been taken from [this repo](https://github.com/cryu854/FastStyle).

We modified the script train.py to export a SavedModel with specific input dimensions. In our experiments the specific input dimensions were necessary to correctly load and execute the model in C++.

Futhermore, the script "checkpoint2SavedModel.py" can convert checkpoints to _SavedModels_ with its input dimensions defined.

***Note***: you can download checkpoints from the orignial repo, using the bash script _ofxTensorFlow2/scripts/download_training_examples.sh_ or [manually](https://cloud.zkm.de/index.php/s/4STL4aG9NNLC9sD)

Please find the original README below...
# Fast Style Transfer in TensorFlow 2 

This is an implementation of Fast-Style-Transfer on Python 3 and Tensorflow 2. 
The neural network is a combination of Gatys' [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), Johnson's [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/), and Ulyanov's [Instance Normalization](https://arxiv.org/abs/1607.08022). 



## Implementation Details

- The **feed-forward network** is roughly the same as described in Johnson, except that batch normalization is replaced with Ulyanov's instance normalization, and the scaling/offset of the output `tanh` layer is slightly different (for better convergence), also use [Resize-convolution layer](https://distill.pub/2016/deconv-checkerboard/) to replace the regular transposed convolution for better upsampling (to avoid checkerboard artifacts)
- The **loss network** used in this implementation follows [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer) , which is similar to the one described in Gatys , using VGG19 instead of VGG16 and typically using "shallower" layers than in Johnson's implementation,  for larger scale style features in transformation (e.g. use `relu1_1` rather than `relu1_2`).

### Training Style Transfer Networks
Use `main.py` to train a new style transfer network.
Training takes 5~6 hours on a GTX 1060 3GB (when batch size is 2).
**Before you run this, you should run `setup.sh` to download the dataset**. 

Example usage:

    python main.py train    \
      --style ./path/to/style/image.jpg \
      --dataset ./path/to/dataset \
      --weights ./path/to/weights \
      --batch 2    

### Evaluating Style Transfer Networks
Use `main.py` to evaluate a style transfer network. 
Evaluation takes 2s per frame(712x474) on a GTX 1060 3GB.  **Models for evaluation are [located here.](https://drive.google.com/drive/folders/1-ywa__KcK4uEEYOzgfeRCpCzP3RJKBwL?usp=sharing)**

Example usage:

    python main.py evaluate    \
      --weights ./path/to/weights \
      --content ./path/to/content/image.jpg(video.mp4)

### Requirements
You will need the following to run the above:
- TensorFlow >= 2.0
- Python 3.7.5, Pillow 7.0.0, Numpy 1.18, Opencv 4.1.2
- If you want to train (and don't want to wait too long):
  - A decent GPU
  - All the required NVIDIA software to run TF on a GPU (cuda, etc)

### Attributions/Thanks
- Some readme/docs formatting was borrowed from Logan Engstrom's [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)
- Some code was borrowed from TensorFlow documentation [Neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer)