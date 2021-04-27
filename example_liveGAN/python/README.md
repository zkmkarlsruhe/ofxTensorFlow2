# DCGAN

Deep Convolutional Generative Adversarial Network

[paper](https://arxiv.org/abs/1511.06434) [project](https://github.com/carpedm20/DCGAN-tensorflow)

![](asset/teaser.png)

![](asset/result.jpg)

## Train

Make a folder under `dataset` and put your images in it, just like the `celeba` folder

Convert the data to tfrecord for convenience, where the default value of `--dataset_name` is `celeba`

```
python main.py --dataset_name your_dataset_name --phase tfrecord
```

Train the model

```
python main.py --dataset_name your_dataset_name --phase train
```

## Test

Test the model

```
python main.py --dataset_name your_dataset_name --phase test
```

Or download the pretrained model of `celeba`, unzip it and you will get a folder named `output`

```
python main.py --phase test
```

The generated image is saved as `output/DCGAN_{dataset_name}/result/result.jpg`