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
	ofSetWindowTitle("example_basics_efficientnet");

	// load the model, bail out on error
	if(!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}

	// define and print image path relative to bin/data
	std::string path(ofToDataPath("my_cat.jpg"));
	ofLog() << "Loading image: " << path;

	// use TensorFlow ops through the cppflow wrappers
	// load a jpeg picture, cast it to float, and add a dimension for batches
	auto input = cppflow::decode_jpeg(cppflow::read_file(path));
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::expand_dims(input, 0);

	// infer the model
	auto output = model.runModel(input);

	// interpret the output
	auto maxLabel = cppflow::arg_max(output, 1);
	ofLog() << "Maximum likelihood: " << maxLabel;

	// access each element using ofxTF2 conversion functions
	ofxTF2::tensorToVector<float>(output, outputVector);
	
	// get and print tensor shape,
	// "NHWC" -> Num_samples x Height x Width x Channels
	auto shape = ofxTF2::getTensorShape(input);
	ofLog() << "Input tensor has shape: "
			<< ofxTF2::vectorToString(shape);
	ofLog() << "Keep in mind: Default format for images in TensorFlow is NHWC";

	// allocate the image and write to it
	imgIn.allocate(shape[2], shape[1], OF_IMAGE_COLOR);
	ofxTF2::tensorToImage<float>(input, imgIn);

	// load a font for displaying strings
	font.load(OF_TTF_SANS, 14);
}

//--------------------------------------------------------------
void ofApp::update() {
	imgIn.update();
}

//--------------------------------------------------------------
void ofApp::draw() {
	font.drawString("[281] tabby cat: " + std::to_string(outputVector[281]), 0, 20);
	font.drawString("[282] tiger cat: " + std::to_string(outputVector[282]), 0, 50);
	font.drawString("[283] persian cat: " + std::to_string(outputVector[283]), 0, 80);
	font.drawString("[284] siamese cat: " + std::to_string(outputVector[284]), 0, 110);
	font.drawString("[285] egyptian cat: " + std::to_string(outputVector[285]), 0, 140);
	imgIn.draw(0, 150);
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
