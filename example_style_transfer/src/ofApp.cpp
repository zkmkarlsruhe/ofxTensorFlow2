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
#include "ofLog.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(30);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_style_transfer");

	ofDirectory modelsDir(ofToDataPath("models"));
	modelsDir.listDir();
	
	// go through the directory and print out all the paths
	for(int i = 0; i < modelsDir.size(); i++){
		ofDirectory sub(modelsDir.getPath(i));
		if (sub.isDirectory()){
			auto absSubPath = sub.getAbsolutePath();
   			ofLogNotice() << "Found model: " << absSubPath;
			modelPaths.push_back(absSubPath);
		}
	}

	model.load(modelPaths[0]);
	modelCounter = 5;
	frameCounter = 0;
	waitNumFrames = 150; 

	nnWidth = 640;
	nnHeight = 480;
	imgIn.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);

#ifdef USE_LIVE_VIDEO
	// try to grab at this size
	camWidth = 640;
	camHeight = 480;
	vidIn.setDesiredFrameRate(30);
	vidIn.setup(camWidth, camHeight);
#endif

	// start the model!
	model.setIdleTime(1);
	model.startThread();
}

//--------------------------------------------------------------
void ofApp::update(){

	// load a new model from the directory every other waitNumFrames frames
	if (frameCounter >= waitNumFrames){
		frameCounter = 0;
		loadNewModel = true;
	}
	if (loadNewModel){
		loadNewModel = false;
		if (modelCounter >= modelPaths.size())
			modelCounter = 0;
		ofLogNotice() << "Load model: " << modelPaths[modelCounter];
		model.load(modelPaths[modelCounter]);
		modelCounter++;
	}

#ifdef USE_LIVE_VIDEO
	// create tensor from video
	vidIn.update();
	if(vidIn.isFrameNew()){	

		// get the frame, resize and copy to tensor
		ofPixels & pixels = vidIn.getPixels();
		ofPixels resizedPixels(pixels);
		resizedPixels.resize(nnWidth, nnHeight);
		input = ofxTF2::pixelsToTensor<float>(resizedPixels);

#else
		// std::string imgPath(ofToDataPath("cat512x512.jpg"));
		std::string imgPath(ofToDataPath("cat640x480.jpg"));
		// create tensor from image file
		input = cppflow::decode_jpeg(cppflow::read_file(imgPath));
		input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
#endif

		// copy input to image
		auto & inputPixels = imgIn.getPixels();
		ofxTF2::tensorToPixels<float>(input, inputPixels);
		imgIn.update();

		// thread-safe conditional input update
		if (model.readyForInput()){
			model.update(input);
		}

		// thread-safe conditional output update
		if (model.isOutputNew()){
			auto output = model.getOutput();
			ofxTF2::tensorToImage<float>(output, imgOut);
			imgOut.update();
		}

#ifdef USE_LIVE_VIDEO
	} // close frame loop
	else{
		// try again later
	}
#endif
	frameCounter++;
}

//--------------------------------------------------------------
void ofApp::draw(){

	int pad = 12;

	ofSetColor(255);
	imgOut.draw(pad, pad, nnWidth*2, nnHeight*2);

	std::string text("Loading new model in ");
	text += std::to_string(waitNumFrames - frameCounter);
	text += " frames";
	ofDrawBitmapString(text, pad, pad);
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
