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

	model = new cppflow::model(ofToDataPath("model"));

	nnWidth = 256;
	nnHeight = 256;
#ifdef USE_LIVE_VIDEO
	// try to grab at this size
	camWidth = 640;
	camHeight = 360;
	vidIn.setDesiredFrameRate(30);
	vidIn.setup(camWidth, camHeight);
#endif
	imgIn.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::update(){

#ifdef USE_LIVE_VIDEO
	// create tensor from video
	vidIn.update();
	if(vidIn.isFrameNew()){

		// get the frame
		ofPixels & pixels = vidIn.getPixels();

		// resize pixels
		ofPixels resizedPixels(pixels);
		resizedPixels.resize(nnWidth, nnHeight);

		// copy to tensor
		input = cppflow::tensor(
			  std::vector<float>(resizedPixels.begin(),
								  resizedPixels.end()),
							  {nnWidth, nnHeight, 3});
	}
	else{
		// try again later
		return;
	}
#else
	// create tensor from image file
	input = cppflow::decode_jpeg(cppflow::read_file(ofToDataPath("cat2.jpg")));
#endif

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

	// ofLog() << output;
	ofLog() << "Time: " << diff.count() << "s Fps: " << ofGetFrameRate();

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
	ofSetColor(255);
	imgOut.draw(0, 0);
	imgIn.draw(0, nnHeight);
#ifdef USE_LIVE_VIDEO
	vidIn.draw(nnWidth, 0, camWidth, camHeight);
#endif
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
#ifdef USE_LIVE_VIDEO
	if(key == 's' || key == 'S'){
		vidIn.videoSettings();
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
