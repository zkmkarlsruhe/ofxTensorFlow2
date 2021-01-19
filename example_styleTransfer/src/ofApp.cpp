#include "ofApp.h"
#include "cppflow/ops.h"
#include "cppflow/model.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_styleTransfer");

	model = new cppflow::model(ofToDataPath("model"));

	nnWidth = 512;
	nnHeight = 512;
	
	imgIn.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::update(){

	// create tensor from image file
	input = cppflow::decode_jpeg(cppflow::read_file(ofToDataPath("cat3.jpg")));
	
	// cast data type and expand to batch size of 1
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::expand_dims(input, 0);

	// start neural network and time measurement
	auto start = std::chrono::system_clock::now();
	output = (*model)(input);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	ofLog() << "Time: " << diff.count() << "s " << ofGetFrameRate() << " fps";

	auto outputVector = output.get_data<float>();
	auto inputVector = input.get_data<float>();

	auto & pixels = imgOut.getPixels();
	for(int i = 0; i < pixels.size(); i++){
		pixels[i] = outputVector[i];
	}

	auto & pixels_in = imgIn.getPixels();
	for(int i = 0; i < pixels_in.size(); i++){
		pixels_in[i] = inputVector[i];
	}

	imgOut.update();
	imgIn.update();
}

//--------------------------------------------------------------
void ofApp::draw(){
	imgIn.draw(0, 0);
	imgOut.draw(nnWidth, nnHeight, nnWidth * 2, nnHeight * 2);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

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
