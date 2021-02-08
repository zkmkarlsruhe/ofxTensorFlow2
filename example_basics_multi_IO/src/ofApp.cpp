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
	ofSetWindowTitle("example_basics_multi_IO");

	// load the model
	model.load("model");

	// setup: define the in and output names
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
	auto inputBody = cppflow::fill({1, 2}, 2.0f);
	auto inputTags = cppflow::fill({1, 12}, 1.0f);
	auto inputTitle = cppflow::fill({1, 3}, 4.0f);

	// convert input tensors to vectors for displaying
	ofxTF2::tensorToVector<float>(inputTitle, titleVector);
	ofxTF2::tensorToVector<float>(inputBody, bodyVector);
	ofxTF2::tensorToVector<float>(inputTags, tagsVector);

	// wrap input tensors in a vector for processing
	std::vector<cppflow::tensor> vectorOfInputTensors = {
		 inputBody, inputTags, inputTitle
	};	

	// inference
	auto vectorOfOutputTensors = model.runMultiModel(vectorOfInputTensors);

	// extract output
	auto outputPrio = vectorOfOutputTensors[0];
	auto outputDept = vectorOfOutputTensors[1];
	ofxTF2::tensorToVector<float>(outputPrio, prioVector);
	ofxTF2::tensorToVector<float>(outputDept, deptVector);

	// load a font for displaying strings
	font.load(OF_TTF_SANS, 14);
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

	font.drawString("Flattened Input titles: ", 20, 20);
	font.drawString(ofxTF2::vectorToString(titleVector), 40, 40);
	font.drawString("Flattened Input body: ", 20, 60);
	font.drawString(ofxTF2::vectorToString(bodyVector), 40, 80);
	font.drawString("Flattened Input tags: ", 20, 100);
	font.drawString(ofxTF2::vectorToString(tagsVector), 40, 120);
	font.drawString("Flattened Output prio:", 20, 150);
	font.drawString(ofxTF2::vectorToString(prioVector), 40, 170);
	font.drawString("Flattened Output dept:", 20, 190);
	font.drawString(ofxTF2::vectorToString(deptVector), 40, 210);
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
