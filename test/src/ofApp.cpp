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
#include "ofxTensorFlow2.h"

#include "testOfxTF2Model.h"
#include "testOfxTF2Tensor.h"
#include "testMemoryLayout.h"


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

	testOfxTF2Tensor();

	testOfxTF2Model(input, output);

	testMemoryLayout();

    ofExit(0);
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

}