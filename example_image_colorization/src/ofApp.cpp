/*
 * Example made with love by Jonathhhan 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */
#include "ofApp.h"
#include "rgbtolab.h" // for color conversion

//--------------------------------------------------------------
// convert float image color space from RGB to LAB
void imageToLAB(ofFloatImage & imgIn) {
	if(imgIn.getImageType() == OF_IMAGE_GRAYSCALE) {
		imgIn.setImageType(OF_IMAGE_COLOR);
	}
	ofFloatPixels & pixels = imgIn.getPixels();
	for(int x = 0; x < pixels.getWidth(); x++) {
		for(int y = 0; y < pixels.getHeight(); y++) {
			ofFloatColor c = pixels.getColor(x, y);
			float l, a, b;
			rgbtolab(c.r * 255.f, c.g * 255.f, c.b * 255.f, &l, &a, &b);
			c.r = l;
			c.g = a;
			c.b = b;
			pixels.setColor(x, y, c);
		}
	}
}

//--------------------------------------------------------------
// convert float image color space from LAB to RGB
void LABtoImage(ofFloatImage & imgIn) {
	ofFloatPixels & pixels = imgIn.getPixels();
	for(int x = 0; x < pixels.getWidth(); x++) {
		for(int y = 0; y < pixels.getHeight(); y++) {
			ofFloatColor c = pixels.getColor(x, y);
			int r, g, b;
			labtorgb(c.r, c.g, c.b, &r, &g, &b);
			c.r = (float)r / 255.f;
			c.g = (float)g / 255.f;
			c.b = (float)b / 255.f;
			pixels.setColor(x, y, c);
		}
	}
}

//--------------------------------------------------------------
cppflow::tensor runInference(cppflow::tensor & input, const ofxTF2::Model & model, int width, int height) {

	// expand and resize the input
	input = cppflow::expand_dims(input, 0);
	auto inputResized = cppflow::resize_bicubic(input, cppflow::tensor({256, 256}), true);

	// compute, scale and resize the remaining channels
	auto output = model.runModel(inputResized);
	output = cppflow::mul(output, cppflow::tensor({128.f}));
	output = cppflow::resize_bicubic(output, cppflow::tensor({height, width}), true);

	// concat the output
	std::vector<cppflow::tensor> inputVector = {input, output};
	return cppflow::concat(cppflow::tensor({3}), inputVector);
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_image_colorization");

	// ofxTF2 setup
	if(!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}
	if(!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	std::vector<std::string> inputNames = {
		"serving_default_input_1",
		"serving_default_placeholder_1"
	};
	std::vector<std::string> outputNames = {
		"StatefulPartitionedCall"
	};
	model.setup(inputNames, outputNames);

#ifdef USE_MOVIE
	// load video and allocate memory
	video.load("sunset_baw.mp4");
	imageWidth = video.getWidth();
	imageHeight = video.getHeight();
	imgOut.allocate(imageWidth, imageHeight, OF_IMAGE_COLOR);
	imgIn.allocate(imageWidth, imageHeight, OF_IMAGE_COLOR);
	video.play();
#else
	// load image and allocate memory
	imgIn.load("wald.jpg");
	imageWidth = imgIn.getWidth();
	imageHeight = imgIn.getHeight();
	imgOut.allocate(imageWidth, imageHeight, OF_IMAGE_COLOR);

	// convert to LAB
	imageToLAB(imgIn);

	// convert the image to pixels
	auto input = ofxTF2::pixelsToTensor(imgIn.getPixels().getChannel(0));

	// compute the colorized image
	auto output = runInference(input, model, imageWidth, imageHeight);

	// convert image
	ofxTF2::tensorToImage(output, imgOut);
	LABtoImage(imgOut);
	imgOut.update();
	imgOut.save("wald_colorized.jpg");
#endif
}

//--------------------------------------------------------------
void ofApp::update() {
#ifdef USE_MOVIE
	video.update();
	if(video.isFrameNew()) {

		// get new frame
		imgIn.setFromPixels(video.getPixels());
		imgIn.update();
		imageToLAB(imgIn); // convert to LAB color space expected by model

		// convert first channel to tensor
		auto input = ofxTF2::pixelsToTensor(imgIn.getPixels().getChannel(0));

		// compute the colorized image
		auto output = runInference(input, model, imageWidth, imageHeight);

		// convert output to RGB
		ofxTF2::tensorToImage(output, imgOut);
		LABtoImage(imgOut); // convert model LAB to RGB space
		imgOut.update();
	}
#endif
}

//--------------------------------------------------------------
void ofApp::draw() {
	imgIn.draw(20, 20, 480, 360);
	imgOut.draw(500, 20, 480, 360);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch(key) {
		case ' ':
			// toggle video playback
			#ifdef USE_MOVIE
				video.setPaused(!video.isPaused());
			#endif
			break;
		case 'r':
			// restart video
			#ifdef USE_MOVIE
				video.stop();
				video.play();
			#endif
			break;
		default: break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}
