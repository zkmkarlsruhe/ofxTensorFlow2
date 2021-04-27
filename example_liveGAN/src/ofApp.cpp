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
#include <stdlib.h>

// computes the dft of real valued inputs
// expects the output vector to be allocated before hand
void dft(const std::vector<float> & audio, std::vector<float> & spec, size_t specSize){
	uint32_t audioSize = audio.size();
	for (float k = 0; k < specSize; k++) {  // For each output element
		float sumreal = 0.0;
		float sumimag = 0.0;
		for (float t = 0; t < audioSize; t++) {  // For each input element
			float angle = 2 * PI * t * k / audioSize;
			sumreal +=  audio[t] * cos(angle);
			sumimag += -audio[t] * sin(angle);
		}
		spec[k] = sqrt(sumreal*sumreal + sumimag*sumimag);
	}
}

// used to remove the mean of the input to the neural network
// this makes sense as the network is trained on noise without mean
void removeMean(std::vector<float> & audio){
	uint32_t audioSize = audio.size();
	float total = 0.0;
	for (int i = 0; i < audioSize; i++) {  
		total += audio[i];
	}
	float mean = total / audioSize;
	for (int i = 0; i < audioSize; i++) {  
		audio[i] -= mean;
	}
}

//--------------------------------------------------------------
void ofApp::setup() {	
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_liveGAN");

	// apply settings to soundStream 
	soundStream.printDeviceList();
	ofSoundStreamSettings settings;
	auto devices = soundStream.getMatchingDevices("default");
	if(!devices.empty()) {
		settings.setInDevice(devices[0]);
	}
	settings.setInListener(this);
	settings.sampleRate = 48000;
	settings.numOutputChannels = 0;
	settings.numInputChannels = 1;
	settings.bufferSize = 512;
	soundStream.setup(settings);

	if(!generator.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	latentSize = 128;

	// allocate spectrum vector
	spec.resize(latentSize);
	curSpec.resize(latentSize);

	// allocate output image
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);

	generator.startThread();
}

//--------------------------------------------------------------
void ofApp::update() {

	// handle new audio frames
	if (newAudio){
		// compute the dft and remove the mean
		dft(audio, curSpec, latentSize);
		removeMean(curSpec);
		// smooth the spectrum
		float alpha = 0.7;
		float beta = 1 - alpha;
		for (size_t i = 0; i < latentSize; i++) {
			spec[i] = alpha * spec[i] + beta * curSpec[i];
		}
		// control signals
		newSpec = true;
		newAudio = false;
	}

	// feed the neural networks thread
	if (newSpec && generator.readyForInput()) {
		// convert spectrum to tensor and add a dim for the batch
		auto spectrumTensor = ofxTF2::vectorToTensor(spec, {1, latentSize});
		generator.update(spectrumTensor);
		newSpec = false;
	}

	// retrieve and handle output
	if (generator.isOutputNew()){
		auto output = generator.getOutput();
		ofxTF2::tensorToImage(output, imgOut);
		imgOut.update();
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	ofSetColor(255);

	// draw output
	imgOut.draw(0, 0, ofGetWidth(), ofGetHeight());

	// draw fps
	std::string text = ofToString((int)ofGetFrameRate()) + " fps";
	ofDrawBitmapStringHighlight(text, ofGetWidth() - 65, 15);
}


//--------------------------------------------------------------
void ofApp::audioIn(ofSoundBuffer & input) {
	audio = input.getBuffer();
	newAudio = true;	
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
void ofApp::dragEvent(ofDragInfo dragInfo) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}
