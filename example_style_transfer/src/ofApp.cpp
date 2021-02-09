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
	ofSetWindowTitle("example_style_transfer");

	// go through the models directory and print out all the paths
	ofDirectory modelsDir(ofToDataPath("models"));
	modelsDir.listDir();
	for(int i = 0; i < modelsDir.size(); i++) {
		ofDirectory sub(modelsDir.getPath(i));
		if(sub.isDirectory()) {
			auto absSubPath = sub.getAbsolutePath();
   			ofLogNotice() << "Found model: " << absSubPath;
			modelPaths.push_back(absSubPath);
		}
	}

	// load first model, bail out on error
	if(!model.load(modelPaths[modelIndex])) {
		std::exit(EXIT_FAILURE);
	}
	modelName = ofFilePath::getBaseName(modelPaths[modelIndex]);

#ifdef USE_LIVE_VIDEO
	// setup video grabber
	vidIn.setDesiredFrameRate(30);
	vidIn.setup(camWidth, camHeight);
#else
	// load input image
	ofImage imgIn;
	//imgIn.load("zkm512x512.jpg"); // alt square image
	imgIn.load("zkm640x480.jpg");
	input = ofxTF2::imageToTensor(imgIn);

	// alternatively, load input image via cppflow
	//std::string imgPath(ofToDataPath("cat512x512.jpg")); // smaller
	//std::string imgPath(ofToDataPath("cat640x480.jpg")); // bigger
	//input = cppflow::decode_jpeg(cppflow::read_file(imgPath));
	//input = cppflow::cast(input, TF_UINT8, TF_FLOAT);

	newInput = true;
#endif

	// allocate output image
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);

	// start the model!
	model.setIdleTime(1); // very short idle time for fast systems
	//model.setIdleTime(33); // longer ~1 fps idle time for slower systems
	model.startThread();
	loadTimestamp = ofGetElapsedTimef();
}

//--------------------------------------------------------------
void ofApp::update() {

	// load a new model after a timeout
	if(autoLoad && ofGetElapsedTimef() - loadTimestamp >= loadTimeSeconds) {
		modelIndex++;
		if(modelIndex >= modelPaths.size()) {
			modelIndex = 0;
		}
		ofLogNotice() << "Load model: " << modelPaths[modelIndex];
		if(!model.load(modelPaths[modelIndex])) {
			// exit gracefully if we can't load model
			ofExit(EXIT_FAILURE);
		}
		modelName = ofFilePath::getBaseName(modelPaths[modelIndex]);
		loadTimestamp = ofGetElapsedTimef();
		newInput = true; // try to update
	}

#ifdef USE_LIVE_VIDEO
	// create tensor from video frame
	vidIn.update();
	if(vidIn.isFrameNew()) {
		// get the frame, resize, and copy to tensor
		ofPixels & pixels = vidIn.getPixels();
		ofPixels resizedPixels(pixels);
		resizedPixels.resize(nnWidth, nnHeight);
		input = ofxTF2::pixelsToTensor(resizedPixels);
		newInput = true;
	}
#else
	// input tensor already created from image file
#endif

	// thread-safe conditional input update
	if(newInput && model.readyForInput()) {
		model.update(input);
	}

	// thread-safe conditional output update
	if(model.isOutputNew()) {
		auto output = model.getOutput();
		ofxTF2::tensorToImage(output, imgOut);
		imgOut.update();
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	ofSetColor(255);

	// draw image
	// TODO: doesn't handle aspect ratio differences...
	imgOut.draw(0, 0, ofGetWidth(), ofGetHeight());

	// draw change info
	float diff = (ofGetElapsedTimef() - loadTimestamp);
	std::string text = "Model: " + modelName;
	if(autoLoad) {
		text += "\nLoading new model in ";
		text += std::to_string((int)(loadTimeSeconds - diff) + 1);
	}
	ofDrawBitmapStringHighlight(text, 4, 12);

	// draw fps
	text = ofToString((int)ofGetFrameRate()) + " fps\n";
	text += "a - toggle auto load";
	ofDrawBitmapStringHighlight(text, ofGetWidth() - 184, 12);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch(key) {
		case 'a': case 'A':
			autoLoad = !autoLoad;
			if(autoLoad) {
				loadTimestamp = ofGetElapsedTimef();
			}
			break;
	}
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
void ofApp::dragEvent(ofDragInfo dragInfo) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}
