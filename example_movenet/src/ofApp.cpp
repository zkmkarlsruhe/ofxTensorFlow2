/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_movenet");

	if(!movenet.setup("model")) {
		std::exit(EXIT_FAILURE);
	}

	#ifdef USE_LIVE_VIDEO
		// setup video grabber
		video.setDesiredFrameRate(30);
		video.setup(camWidth, camHeight);
		imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
	#else
		video.load("production ID 3873059_2.mp4");
		video.play();
	#endif
}

//--------------------------------------------------------------
void ofApp::update() {
	video.update();
	if(video.isFrameNew()) {
		ofPixels pixels(video.getPixels());
		#ifdef USE_LIVE_VIDEO
			pixels.resize(nnWidth, nnHeight);
			imgOut.setFromPixels(pixels);
			imgOut.update();
		#endif
		movenet.update(pixels);
	}
}

//--------------------------------------------------------------
void ofApp::draw() {

	#ifdef USE_LIVE_VIDEO
		imgOut.draw(0,0);
	#else
		video.draw(0,0);
	#endif

	movenet.draw();

	ofDrawBitmapStringHighlight(ofToString((int)ofGetFrameRate()) + " fps", 4, 12);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	#ifndef USE_LIVE_VIDEO
		if(key == 'r') {
			video.stop();
			video.play();
		}
	#endif
}
