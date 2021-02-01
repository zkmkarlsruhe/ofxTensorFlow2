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
	ofSetWindowTitle("example_pix2pix");

	model.load("model");

	nnWidth = 256;
	nnHeight = 256;
#ifdef USE_LIVE_VIDEO
	// try to grab at this size
	camWidth = 640;
	camHeight = 360;
	vidSrc.setDesiredFrameRate(30);
	vidSrc.setup(camWidth, camHeight);
#else
	// note: model expects RGB only, no alpha!
	imgSrc.load("shoe.png");
	if(imgSrc.getWidth() != nnWidth || imgSrc.getHeight() != nnHeight) {
		ofLog() << "resizing source to " << nnWidth << "x" << nnHeight;
		imgSrc.resize(nnWidth, nnHeight);
	}
#endif
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);

	// shorten idle time to have model check for input more frequently,
	// this may increase responsivity on faster machines but will use more cpu
	model.setIdleTime(10);

	// start the model background thread
	model.startThread();
}

//--------------------------------------------------------------
void ofApp::update(){
	bool newInput = false;

#ifdef USE_LIVE_VIDEO
	// create tensor from video
	vidSrc.update();
	if(vidSrc.isFrameNew() && model.readyForInput()){

		// get the frame
		ofPixels & pixels = vidSrc.getPixels();

		// resize pixels
		ofPixels resizedPixels(pixels);
		resizedPixels.resize(nnWidth, nnHeight);

		// copy to tensor
		input = ofxTF2::pixelsToTensor<float>(resizedPixels);
		newInput = true;
	}
#else
	if(model.readyForInput()) {
		// create tensor from image
		input = ofxTF2::imageToTensor<float>(imgSrc);
		newInput = true;
	}
#endif

	// process any input
	if(newInput) {

		// end measuremnt
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		ofLog() << "run took: " << diff.count() << " s or ~" << (int)(1.0/diff.count()) << " fps";

		// feed input into model
		model.update(input);
	}

	// process any output
	if(model.isOutputNew()) {

		// pull output from model
		output = model.getOutput();

		// start new measurement
		start = std::chrono::system_clock::now();

		ofxTF2::tensorToImage<float>(output, imgOut);
		imgOut.update();
	}
}

//--------------------------------------------------------------
void ofApp::draw(){

	ofSetColor(255);
	imgOut.draw(12, 12);
#ifdef USE_LIVE_VIDEO
	vidSrc.draw(12 + nnWidth, 12, camWidth, camHeight);
	ofColor(0);
#else
	imgSrc.draw(12 + nnWidth, 12);
#endif

	ofSetColor(255);
	ofDrawBitmapString("output", 12, 10);
//	ofDrawBitmapString("input", 12, 24 + nnHeight);
	ofDrawBitmapString("source", 12 + nnWidth, 10);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
#ifdef USE_LIVE_VIDEO
	if(key == 's' || key == 'S'){
		vidSrc.videoSettings();
	}
#endif
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
