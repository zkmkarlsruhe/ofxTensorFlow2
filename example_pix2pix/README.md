# Pix2Pix
This example explores image generation using neural networks and is built around Memo Akten's Pix2Pix example of [ofxMSATensorFlow](https://github.com/memo/ofxMSATensorFlow). The image to image translation is implemented using a Generative Adversarial Network (GAN).

A basic GAN consists of two parts a generator that takes in an input and generates a desired output (here: in both cases images) and a classifier that tries to predict if its input was generated or not. While the classifier is trained in a classic manner on real and fake samples the generator is trained _through_ the classifier. That is, its update depends on the output of the classifier when given a newly generated sample. A training step includes training both parts side by side.

### TensorFlow2
As with all examples you can download the pretrained model from the assests, copy it to the bin/data folder and name it model.

If you want to train it by yourself you can edit the config.py file in the python folder and run main.py. It is highly recommended to use a GPU for this purpose.

Check this [post](https://www.tensorflow.org/tutorials/generative/pix2pix?hl=en) for more information on the training procedure.

### openFrameworks
In this example we are using the a specification of the `ofxTF2::Model`. It allows to run the inference of the model asynchronous to the rest of the program. This is especially helpful as it's not delaying other tasks such as drawing. The `ofxTF2::ThreadedModel` uses the following pattern.

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
In this example we use this pattern but also define a subclass with a specialized runModel function to augment the asynchronous model execution with pre- and postprocessing. This approach is further explained in the style transfer example.

However, it is very important to reconstruct the way data is treated during the training, which does not include ways that enhance generalization such as noise and image augmentation.

Once more we want to stress that you can call many TensorFlow operations through cppflow. 
```C++
input = cppflow::div(input, cppflow::tensor({127.5f}));
input = cppflow::add(input, cppflow::tensor({-1.0f}));
```
