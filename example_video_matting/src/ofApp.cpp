/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_video_matting");

	#ifdef USE_LIVE_VIDEO
		// setup video grabber
		video.setDesiredFrameRate(30);
		video.setup(camWidth, camHeight);
	#else
		video.load("codylexi.mp4");
		video.play();
	#endif

	if(!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}

	std::vector<std::string> inputNames = {
		"serving_default_downsample_ratio:0",
		"serving_default_r1i:0",
		"serving_default_r2i:0",
		"serving_default_r3i:0",
		"serving_default_r4i:0",
		"serving_default_src:0"
	};
	std::vector<std::string> outputNames = {
		"StatefulPartitionedCall:0",
		"StatefulPartitionedCall:1",
		"StatefulPartitionedCall:2",
		"StatefulPartitionedCall:3",
		"StatefulPartitionedCall:4",
		"StatefulPartitionedCall:5"
	};
	model.setup(inputNames, outputNames);

	// parameters for the neural network
	float downsampleRatio = 0.25f;
	float videoWidth = video.getWidth();
	float videoHeight = video.getHeight();
	float batchSize = 1.0f;
	float numChannels = 3.0f;
	
	// model-specific inputs
	inputs = {
		cppflow::tensor({downsampleRatio}),
		cppflow::tensor({0.0f}),                         // r1i
		cppflow::tensor({0.0f}),                         // r2i
		cppflow::tensor({0.0f}),                         // r3i
		cppflow::tensor({0.0f}),                         // r4i
		cppflow::tensor({batchSize, videoHeight, videoWidth, numChannels})
	};

	imgMask.allocate(video.getWidth(), video.getHeight(), OF_IMAGE_GRAYSCALE);

	imgOut.allocate(video.getWidth(), video.getHeight(), OF_IMAGE_COLOR_ALPHA);
	imgOut.getTexture().setAlphaMask(imgMask.getTexture());

	imgBackground.load("bg.jpg");
}

//--------------------------------------------------------------
void ofApp::update() {

	video.update();
	if(video.isFrameNew()) {
		ofPixels & pixels = video.getPixels();

		// prepare inputs
		auto input = ofxTF2::pixelsToTensor(pixels);
		auto inputCast = cppflow::cast(input, TF_UINT8, TF_FLOAT);
		inputCast = cppflow::mul(inputCast, cppflow::tensor({1/255.0f}));
		inputCast = cppflow::expand_dims(inputCast, 0);
		inputs[5] = inputCast;

		// run model
		auto outputs = model.runMultiModel(inputs);

		// process outputs
		inputs[1] = outputs[2];
		inputs[2] = outputs[3];
		inputs[3] = outputs[4];
		inputs[4] = outputs[5];
		auto foreground = outputs[1];
		foreground = cppflow::mul(foreground, cppflow::tensor({255.0f}));
		auto foregroundMod = cppflow::cast(foreground, TF_FLOAT, TF_UINT8);
		ofxTF2::tensorToImage(foreground, imgMask);
		imgMask.update();
		imgOut.setFromPixels(video.getPixels());
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	float w = ofGetWidth() / 2;
	float h = ofGetHeight() / 2;

	// row 1
	video.draw(0, 0, w, h);
	imgMask.draw(w, 0, w, h);

	// row 2
	imgBackground.draw(w/2, h, w, h);
	imgOut.draw(w/2, h, w, h);

	ofDrawBitmapStringHighlight(ofToString((int)ofGetFrameRate()) + " fps", 4, 12);
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
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}
