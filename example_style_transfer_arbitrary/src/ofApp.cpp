#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(10);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_style_transfer_arbitrary");

	videoPlayer.load("Movie.mp4");
	videoPlayer.play();
	imgOut.allocate(resultWidth, resultHeight, OF_IMAGE_COLOR);

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_90, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}
	// load first model, bail out on error
	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({ {"serving_default_placeholder"} ,{"serving_default_placeholder_1"} },  {"StatefulPartitionedCall"} );

	// load style image
	style = cppflow::decode_jpeg(cppflow::read_file(std::string(ofToDataPath("wave.jpg"))));
	style = cppflow::expand_dims(style, 0);
	style = cppflow::resize_bicubic(style, cppflow::tensor({ 256, 256 }), true);
	style = cppflow::cast(style, TF_UINT8, TF_FLOAT);
	style = cppflow::mul(style, cppflow::tensor({ 1.0f / 255.f }));
	inputVector = { style, style };
}

//--------------------------------------------------------------
void ofApp::update() {
	videoPlayer.update();
	if (videoPlayer.isFrameNew()) {
		videoPlayer.getTexture().readToPixels(floatPixels);
		input = ofxTF2::pixelsToTensor(floatPixels);
		input = cppflow::expand_dims(input, 0);
		if (resultHeight != videoPlayer.getHeight() || resultWidth != videoPlayer.getWidth()){
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