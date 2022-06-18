#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_style_transfer");
	videoPlayer.load("Frenzy.mp4");
	videoPlayer.play();
	
	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}
	// load first model, bail out on error
	if(!model.load("magenta")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({ {"serving_default_placeholder"} ,{"serving_default_placeholder_1"} }, { {"StatefulPartitionedCall"} });
	input = cppflow::decode_jpeg(cppflow::read_file(std::string(ofToDataPath("wave.jpg"))));
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = input / 255.f;
	input = cppflow::expand_dims(input, 0);
}

//--------------------------------------------------------------
void ofApp::update() {
	videoPlayer.update();
	if (videoPlayer.isFrameNew()) {
		ofPixels& pixels = videoPlayer.getPixels();
		auto input2 = ofxTF2::pixelsToTensor(pixels);
		input2 = cppflow::cast(input2, TF_UINT8, TF_FLOAT);
		input2 = input2 / 255.f;
		input2 = cppflow::expand_dims(input2, 0);
		std::vector<cppflow::tensor> vectorOfInputTensors = {
		 input2, input
		};
		auto output = model.runMultiModel(vectorOfInputTensors);
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