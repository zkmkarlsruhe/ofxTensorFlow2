/*
 * ofxTensorFlow2
 *
 * Copyright (c) 2021 ZKM | Hertz-Lab
 * Paul Bethge <bethge@zkm.de>
 * Dan Wilcox <dan.wilcox@zkm.de>
 *
 * BSD Simplified License.
 * For information on usage and redistribution, and for a DISCLAIMER OF ALL
 * WARRANTIES, see the file, "LICENSE.txt," in this distribution.
 *
 * This code has been developed at ZKM | Hertz-Lab as part of „The Intelligent 
 * Museum“ generously funded by the German Federal Cultural Foundation.
 */

#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_effnet");

	// load and infer the model
	ofxTF2Model model("model");

	std::string path(ofToDataPath("my_cat.jpg"));
	ofLog() << "Loading image: " << path;

	// use TensorFlow ops through the cppflow wrappers
	// load a jpeg picture cast it to float and add a dimension for batches
	auto input = cppflow::decode_jpeg(cppflow::read_file(path));
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::expand_dims(input, 0);

	auto output = model.runModel(input);

	// interpret the output
	auto maxLabel = cppflow::arg_max(output, 1);
	ofLog() << "Maximum likelihood: " << maxLabel;

	// access each element via the internal vector
	std::vector<float> outputVector;
	ofxTF2::tensorToVector<float>(output, outputVector);
	
	ofLog() << "[281] tabby cat: " << outputVector[281];
	ofLog() << "[282] tiger cat: " << outputVector[282];
	ofLog() << "[283] persian cat: " << outputVector[283];
	ofLog() << "[284] siamese cat: " << outputVector[284];
	ofLog() << "[285] egyptian cat: " << outputVector[285];

	ofExit();
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
