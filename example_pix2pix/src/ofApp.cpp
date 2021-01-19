#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_pix2pix");

	model = new cppflow::model(ofToDataPath("model"));

	nnWidth = 256;
	nnHeight = 256;
	imgIn.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::update(){

	// create tensor from image file
	input = cppflow::decode_jpeg(cppflow::read_file(ofToDataPath("cat2.jpg")));

	// cast data type and expand to batch size of 1
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::expand_dims(input, 0);

	// apply preprocessing as in python
	input = cppflow::div(input, cppflow::tensor({127.5f}));
	input = cppflow::add(input, cppflow::tensor({-1.0f}));

	// start neural network and time measurement
	auto start = std::chrono::system_clock::now();
	output = (*model)(input);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	// ofLogNotice() << output;
	ofLogNotice() << "Time: " << diff.count() << "s " << ofGetFrameRate() << " fps";

	// copy output to image
	auto outputVector = output.get_data<float>();
	auto & pixels = imgOut.getPixels();
	for(int i = 0; i < pixels.size(); i++){
		pixels[i] = (outputVector[i] + 1) * 127.5;
	}

	// copy input to image
	auto inputVector = input.get_data<float>();
	auto & inputPixels = imgIn.getPixels();
	for(int i = 0; i < inputPixels.size(); i++){
		inputPixels[i] = inputVector[i];
	}

	imgOut.update();
	imgIn.update();
}

//--------------------------------------------------------------
void ofApp::draw(){
	imgOut.draw(0, 0);
	imgIn.draw(0, nnHeight);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
