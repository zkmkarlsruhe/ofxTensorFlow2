# Pix2Pix

The code has been taken from [this repo](https://github.com/soumik12345/Pix2Pix) and received minor changes.

***Note***: the original jupyter notebook to train on facades has been converted and renamed to main.py.
***Note***: the original config.py and source files have been extended with the ability to switch the tasks (i.e. edges <-> shoes), to save models after a certain number of epochs and to export the files as SavedModels.

Please find an excerpt of the original README below...

## Pix2Pix

Tensorflow 2.0 Implementation of the paper [Image-to-Image Translation using Conditional GANs](https://arxiv.org/abs/1611.07004) by [Philip Isola](https://arxiv.org/search/cs?searchtype=author&query=Isola%2C+P), [Jun-Yan Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J), [Tinghui Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+T) and [Alexei A. Efros](https://arxiv.org/search/cs?searchtype=author&query=Efros%2C+A+A).

## Architecture

### Generator

- The Generator is a Unet-Like model with skip connections between encoder and decoder.
- Encoder Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```
- Decode Blocks is ```Conv2DTranspose -> BatchNormalization -> Dropout (optional) -> Activation (ReLU)```

### Discriminator

- PatchGAN Discriminator
- Discriminator Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```


## Loss Functions

### Generator Loss

The Loss function can also be boiled down to

```Loss = GAN_Loss + Lambda * L1_Loss```, where GAN_Loss is Sigmoid Cross Entropy Loss and Lambda = 100 (determined by the authors)

### Discriminator Loss

The Discriminator Loss function can be written as

```Loss = disc_loss(real_images, array of ones) + disc_loss(generated_images, array of zeros)```

where `disc_loss` is Sigmoid Cross Entropy Loss.

## References

All the sources cited during building this codebase are mentioned below:

- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
- [https://github.com/phillipi/pix2pix](https://github.com/phillipi/pix2pix)
- [Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](https://arxiv.org/abs/1604.04382)
- [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
- [Tensorflow Pix2Pix](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)
- [Keras Pix2Pix](https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py)