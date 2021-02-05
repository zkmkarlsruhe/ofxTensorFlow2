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
	ofSetWindowTitle("example_basics");

	// declare and load the model
	ofxTF2::Model model;
	model.load("model");

	// setup: define the in and ouput names
	std::vector<std::string> inputNames = {
		"serving_default_body",
		"serving_default_tags",
		"serving_default_title"
	};
	std::vector<std::string> outputNames = {
		"StatefulPartitionedCall:0",
		"StatefulPartitionedCall:1"
	};
	model.setup(inputNames, outputNames);

	// create tensors for each input
	cppflow::tensor input_title = cppflow::fill({1, 10}, 4.0f);
	cppflow::tensor input_body = cppflow::fill({1, 100}, 2.0f);
	cppflow::tensor input_tags = cppflow::fill({1, 12}, 1.0f);

	// wrap tensors in a vector
	std::vector<cppflow::tensor> vectorOfInputTensors {
		 input_body, input_tags, input_title };

	// inference
	auto vectorOfOutputTensors = model.runMultiModel(vectorOfInputTensors);
	
	// extract ouput
	auto outputPrio = vectorOfOutputTensors[0];
	auto outputDept = vectorOfOutputTensors[1];
	std::vector<float> prioVector;
	std::vector<float> deptVector;
	ofxTF2::tensorToVector<float>(outputPrio, prioVector);
	ofxTF2::tensorToVector<float>(outputDept, deptVector);

	// print
	ofLog() << ofxTF2::vectorToString(prioVector);
	ofLog() << ofxTF2::vectorToString(deptVector);
	
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
