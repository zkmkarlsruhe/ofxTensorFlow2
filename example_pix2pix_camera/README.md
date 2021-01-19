# Pix2pix
This example explores image generation using neural networks. 

### python
Download the pretrained model from the assests, copy it to the bin/data folder and name it model.

If you want to train it by yourself you can edit the config.py file in the python folder. Then run main.py. It is highly recommended to use a graphic card for this purpose.

### openframeworks

Again it is very important to reconstruct the way data is treated during the training, which does not include ways that enhance generalization such as noise and image augmentation.
```c++
input = cppflow::div(input, cppflow::tensor({127.5f}));
input = cppflow::add(input, cppflow::tensor({-1.0f}));
```