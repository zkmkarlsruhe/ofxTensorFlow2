#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle(ofToString(ofGetFrameRate()));
	videoPlayer.load("Frenzy.mp4");
	videoPlayer.play();

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}
	// load first model, bail out on error
	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({ {"serving_default_placeholder"} ,{"serving_default_placeholder_1"} }, { {"StatefulPartitionedCall"} });
	style = cppflow::decode_jpeg(cppflow::read_file(std::string(ofToDataPath("wave.jpg"))));
	style = cppflow::expand_dims(style, 0);
	style = cppflow::resize_bicubic(style, cppflow::tensor({ 256, 256 }), true);
	style = cppflow::cast(style, TF_UINT8, TF_FLOAT);
	style = cppflow::div(style, cppflow::tensor({ 255.f }));
}

//--------------------------------------------------------------
void ofApp::update() {
	videoPlayer.update();
	if (videoPlayer.isFrameNew()) {
		input = ofxTF2::pixelsToTensor(videoPlayer.getPixels());
		input = cppflow::expand_dims(input, 0);
		input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
		input = cppflow::div(input, cppflow::tensor({ 255.f }));
		inputVector = {input, style};
		output = model.runMultiModel(inputVector);
		shape = ofxTF2::getTensorShape(output[0]);
		imgOut.allocate(shape[2], shape[1], OF_IMAGE_COLOR);
		ofxTF2::tensorToImage(output[0], imgOut);
		imgOut.update();
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	imgOut.draw(0, 0, shape[2], shape[1]);
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
