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
	imgIn.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::update(){

#ifdef USE_LIVE_VIDEO
	// create tensor from video
	vidSrc.update();
	if(vidSrc.isFrameNew()){

		// get the frame
		ofPixels & pixels = vidSrc.getPixels();

		// resize pixels
		ofPixels resizedPixels(pixels);
		resizedPixels.resize(nnWidth, nnHeight);

		// copy to tensor
		input = cppflow::pixels_to_tensor(resizedPixels);
	}
	else{
		// try again later
		return;
	}
#else
	// create tensor from image file
	input = cppflow::image_to_tensor(imgSrc);
#endif

	// cast data type and expand to batch size of 1
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::expand_dims(input, 0);

	// apply preprocessing as in python to change range to -1 to 1
	input = cppflow::div(input, cppflow::tensor({127.5f}));
	input = cppflow::sub(input, cppflow::tensor({1.0f}));
	// the above can also be written using math operators:
	//input = (input / cppflow::tensor({127.5f})) - cppflow::tensor({1.0f});

	// start neural network and time measurement
	auto start = std::chrono::system_clock::now();
	output = model.run(input);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	// ofLog() << output;
	ofLog() << "run took: " << diff.count() << "s, fps: " << ofGetFrameRate();

	// postprocess to change range to -1 to 1 and copy output to image
	output = cppflow::add(output, cppflow::tensor({1.0f}));
	output = cppflow::mul(output, cppflow::tensor({127.5f}));
	// the above can also be written using math operators:
	//output = (output + cppflow::tensor({1.0f})) * cppflow::tensor({127.5f});
	cppflow::tensor_to_image(output, imgOut);

//	// postprocess and copy input to image
//	input = cppflow::add(input, cppflow::tensor({1.0f}));
//	input = cppflow::mul(input, cppflow::tensor({127.5f}));
//	cppflow::tensor_to_image(input, imgIn);

	imgOut.update();
	imgIn.update();
}

//--------------------------------------------------------------
void ofApp::draw(){

	ofSetColor(255);
	imgOut.draw(12, 12);
//	imgIn.draw(12, 12 + nnHeight + 12);
#ifdef USE_LIVE_VIDEO
	vidSrc.draw(12 + nnWidth, 12, camWidth, camHeight);
	ofColor(0);
#else
	imgSrc.draw(12 + nnWidth, 12);
#endif

	ofSetColor(255);
	ofDrawBitmapString("output", 12, 8);
//	ofDrawBitmapString("input", 12, 24 + nnHeight);
	ofDrawBitmapString("source", 12 + nnWidth, 8);
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
