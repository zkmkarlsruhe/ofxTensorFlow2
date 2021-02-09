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
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_basics");

	// create an input tensor of an arbitrary shape and fill it
	auto input = cppflow::fill({1, 2, 2, 3}, 1.0f);

	// load the model, bail out on error
	if(!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}

	// inference
	auto output = model.runModel(input);

	// use the ofxTF2 namespace for some useful functions like conversion
	ofxTF2::tensorToVector<float>(output, outputVector);
	ofxTF2::tensorToVector<float>(input, inputVector);

	// print summary to console
	ofLog() << "Flatted Input:";
	ofLog() << ofxTF2::vectorToString(inputVector);
	ofLog() << "Flattened Output:";
	ofLog() << ofxTF2::vectorToString(outputVector);

	// load a font for displaying strings
	font.load(OF_TTF_SANS, 14);
}

//--------------------------------------------------------------
void ofApp::update() {

}

//--------------------------------------------------------------
void ofApp::draw() {

	// draw summary to screen
	font.drawString("Flattened Input:", 20, 20);
	font.drawString(ofxTF2::vectorToString(inputVector), 40, 40);
	font.drawString("Flattened Output:", 20, 60);
	font.drawString(ofxTF2::vectorToString(outputVector), 40, 80);
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
