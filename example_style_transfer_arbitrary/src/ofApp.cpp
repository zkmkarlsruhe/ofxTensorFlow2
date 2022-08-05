/*
 * Example made with love by Jonathhhan 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */
#include "ofApp.h"

//--------------------------------------------------------------
template <typename T>
cppflow::tensor pixelsToFloatTensor(const ofPixels_<T> & pixels) {
	auto t = ofxTF2::pixelsToTensor(pixels);
	t = cppflow::expand_dims(t, 0);
	t = cppflow::cast(t, TF_UINT8, TF_FLOAT);
	t = cppflow::mul(t, cppflow::tensor({1.0f / 255.f}));
	return t;
}

//--------------------------------------------------------------
template <typename T>
cppflow::tensor imageToFloatTensor(const ofImage_<T> & image) {
	return pixelsToFloatTensor(image.getPixels());
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_style_transfer_arbitrary");

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_90, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}
	// load first model, bail out on error
	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}

	// setup: define the input and output names
	std::vector<std::string> inputNames = {
		"serving_default_placeholder",
		"serving_default_placeholder_1"
	};
	std::vector<std::string> outputNames = {
		"StatefulPartitionedCall"
	};
	model.setup(inputNames, outputNames);

	// input style image
	style = imageToFloatTensor(ofImage("wave.jpg"));
	style = cppflow::resize_bicubic(style, cppflow::tensor({256, 256}), true);
	inputVector = {style, style};

	// input video
	videoPlayer.load("movie.mp4");
	videoPlayer.play();

	// output image
	imgOut.allocate(resultWidth, resultHeight, OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::update() {
	videoPlayer.update();
	if(videoPlayer.isFrameNew()) {
		input = pixelsToFloatTensor(videoPlayer.getPixels());
		if(resultHeight != videoPlayer.getHeight() || resultWidth != videoPlayer.getWidth()) {
			input = cppflow::resize_bicubic(input, cppflow::tensor({resultHeight, resultWidth}), true);
		}
		inputVector[0] = input;
		output = model.runMultiModel(inputVector);
		ofxTF2::tensorToImage(output[0], imgOut);
		imgOut.update();
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	imgOut.draw(20, 20, 480, 360);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

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
