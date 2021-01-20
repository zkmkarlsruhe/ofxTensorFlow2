#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("test");


	// ==== Reference === //

	// create a tensor of an arbitrary shape and fill it
	auto input = cppflow::fill({1, 2, 2, 3}, 0.9f);
	// load the cppflow::model
	cppflow::model model(ofToDataPath("model"));
	// inference
	auto output = model(input);
	// print the tensor
	ofLog() << output;

	// ====== Wrapper ====== //
	// ofxTensor from cppflow::tensor
	ofxTensor fromCppflowTensor (input);

	// todo ofxTensor from vector
	// todo ofxTensor from ofImage
	// todo ofxTensor from ofPixels

	// vector from ofxTensor
	auto vecFromOfxTensor = fromCppflowTensor.getData<float>();

	// todo vector from ofxTensor
	// todo ofImage from ofxTensor  
	// todo ofPixels from ofxTensor

	// std::cout << "vecFromOfxTensor size: " << vecFromOfxTensor.size() << std::endl;
	// std::cout << "vecFromCppflowTensor size: " << input.get_data<float>().size() << std::endl;

	for (int i =0; i < vecFromOfxTensor.size(); i++){
		std::cout << vecFromOfxTensor[i] << ", ";
	}
	std::cout.flush();


	// load the ofxModel
	ofxModel simpleModel(ofToDataPath("model"));
	
	// reload
	simpleModel.load(ofToDataPath("model"));

	auto simpleModelOutput = simpleModel.run(input);


}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

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
