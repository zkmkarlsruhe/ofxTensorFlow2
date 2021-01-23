# Pix2Pix
This example explores image generation using neural networks. Specifically Pix2Pix does image to image translation using Generative Adversarial Networks (GAN).

A basic GAN consists of two parts a generator that takes in an input and generates a desired output (here: in both cases images) and a classifier that tries to predict if its input was generated or not. While the classifier is trained in a classic manner on real and fake samples the generator is trained _through_ the classifier. That is, its update depends on the output of the classifier when given a newly generated sample. A training step includes training both parts side by side.

### TensorFlow2
As with all examples you can download the pretrained model from the assests, copy it to the bin/data folder and name it model.

If you want to train it by yourself you can edit the config.py file in the python folder. Then run main.py. It is highly recommended to use a graphic card for this purpose.

Check this [post](https://www.tensorflow.org/tutorials/generative/pix2pix?hl=en) for more information on the trainig procedure.

### openFrameworks
Again it is very important to reconstruct the way data is treated during the training, which does not include ways that enhance generalization such as noise and image augmentation.
```c++
input = cppflow::div(input, cppflow::tensor({127.5f}));
input = cppflow::add(input, cppflow::tensor({-1.0f}));
```
