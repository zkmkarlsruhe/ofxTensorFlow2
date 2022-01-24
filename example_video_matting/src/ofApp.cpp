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

	auto r1i = cppflow::tensor({0.0f});
	auto r2i = cppflow::tensor({0.0f});
	auto r3i = cppflow::tensor({0.0f});
	auto r4i = cppflow::tensor({0.0f});
	auto src = cppflow::tensor({1.0f, nnHeight, nnWidth, 3.0f});
	auto downsample_ratio = cppflow::tensor({0.25f});

	vectorOfInputTensors.push_back(downsample_ratio);
	vectorOfInputTensors.push_back(r1i);
	vectorOfInputTensors.push_back(r2i);
	vectorOfInputTensors.push_back(r3i);
	vectorOfInputTensors.push_back(r4i);
	vectorOfInputTensors.push_back(src);

	mask.allocate(video.getWidth(), video.getHeight(), OF_IMAGE_GRAYSCALE);

	outputMasked.allocate(video.getWidth(), video.getHeight(), OF_IMAGE_COLOR_ALPHA);
	outputMasked.getTexture().setAlphaMask(mask.getTexture());
}

//--------------------------------------------------------------
void ofApp::update() {
	video.update();

	if(video.isFrameNew()) {
		ofPixels & pixels = video.getPixels();
		auto inputpxs = ofxTF2::pixelsToTensor(pixels);
		auto inputCast = cppflow::cast(inputpxs, TF_UINT8, TF_FLOAT);
		inputCast = cppflow::mul(inputCast, cppflow::tensor({1/255.0f}));
		inputCast = cppflow::expand_dims(inputCast, 0);

		vectorOfInputTensors[5] = inputCast;
		
		auto vectorOfOutputTensors = model.runMultiModel(vectorOfInputTensors);

		vectorOfInputTensors[1] = vectorOfOutputTensors[2];
		vectorOfInputTensors[2] = vectorOfOutputTensors[3];
		vectorOfInputTensors[3] = vectorOfOutputTensors[4];
		vectorOfInputTensors[4] = vectorOfOutputTensors[5];

		auto foreground = vectorOfOutputTensors[1];
		foreground = cppflow::mul(foreground, cppflow::tensor({255.0f}));

		auto foregroundMod = cppflow::cast(foreground, TF_FLOAT, TF_UINT8);
		ofxTF2::tensorToImage(foreground, mask);
		mask.update();

		outputMasked.setFromPixels(video.getPixels());
	}
}

//--------------------------------------------------------------
void ofApp::draw() {

	video.draw(0, 0, video.getWidth()/2, video.getHeight()/2);
	mask.draw( video.getWidth()/2,0, video.getWidth()/2, video.getHeight()/2);

	bg.draw(0,video.getHeight()/2, video.getWidth()/2, video.getHeight()/2);
	outputMasked.draw(0,video.getHeight()/2, video.getWidth()/2, video.getHeight()/2);
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
