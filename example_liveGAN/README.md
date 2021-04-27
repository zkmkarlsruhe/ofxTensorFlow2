# liveGAN
This example explores image generation using neural networks. The audio to image translation is implemented using a Generative Adversarial Network (GAN).

A basic GAN consists of two parts a generator that takes in an input and generates a desired output and a classifier that tries to predict if its input was generated or not. In this case, the generator generates an image from a one-dimensional vector of gaussian noise. While the classifier is trained in a classic manner on real and fake samples the generator is trained _through_ the classifier. That is, its update depends on the output of the classifier when given a newly generated sample. A training step includes training both parts side by side.

### TensorFlow2
If you want to train the neural network on your own dataset you simply drop your images in a subfolder inside `dataset/`, create a tfrecord file:
```bash
python3 main.py --dataset_name your_dataset_name --phase tfrecord
```
and start the training process.
```bash
python3 main.py --dataset_name your_dataset_name --phase train
```
The model is set up to output 256x256x3 images. If your output should be different, take a look a `model.py`. Please also take a look at the arguments of `main.py` for more information on the training process.
It is highly recommended to use a GPU for training, however for inference a strong CPU is sufficient. The dataset should consist of 1 to 100 thousand images, depending on the variation.

### openFrameworks
In this example we are using a derivative of the `ofxTF2::Model`. It allows to run the inference of the model asynchronous to the rest of the program. This is especially helpful as it's not delaying other tasks such as drawing. The `ofxTF2::ThreadedModel` uses the following pattern.

Declare the model as a member of the ofApp in ofApp.h:
```c++
class ofApp : public ofBaseApp {
	public:
		//...
		void setup();
		void update();
	private:
		ofxTF2::ThreadedModel model;
		//...
};
```
Utilize the model in ofApp.cpp
```c++
void ofApp::setup() {
	// load the model and start the thread
	model.load("path/to/modeldir");
	model.startThread();
	//...
}

void ofApp::update() {
	// check if the model is ready to receive a new input tensor
	if(model.readyForInput()) {
		// create a tensor and feed it to the network
		cppflow::tensor input = returnTensor();
		model.update(input);
	}
	//...
	// check if the model is done with computation
	if(model.isOutputNew()) {
		// receive and process the output of the network
		cppflow::tensor output = model.getOutput();
		doSomething(output);
	}
	//...
}
// ...
```
In this example we use this pattern but also define a subclass with a specialized runModel function to augment the asynchronous model execution with pre- and postprocessing. This approach is further explained in the style transfer example. An easy way to get the data in the right range is to use the map function `ofxTF2::mapTensorValues()`:
```c++
auto image_tensor = ofxTF2::mapTensorValues(model_output, -1, 1, 0, 255);
```
