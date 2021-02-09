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
 * 
 * This code is based on Memo Akten's ofxMSATensorFlow example.
 */

#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofBackground(54, 54, 54);
	ofSetWindowTitle("example_pix2pix");
	ofSetLogLevel("ofxTensorFlow2", OF_LOG_VERBOSE);

	// neural network setup, bail out on error
	// the default model is edges2shoes and excepts [None, None, None, 3]
	if(!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	
	// allocate fbo and images with correct dimensions, no alpha channel
	ofLogVerbose() << "Allocating fbo and images ("
	               << nnWidth << ", " << nnHeight << ")";
	fbo.allocate(nnWidth, nnHeight, GL_RGB);
	imgIn.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);

	// search for the drawing tool config files in bin/data/draw
	setupDrawingTool("draw");

	// shorten idle time to have model check for input more frequently,
	// this may increase responsiveness on faster machines but will use more cpu
	model.setIdleTime(10);

	// start the model background thread
	model.startThread();
}

void ofApp::update() {

	// start & stop the model
	if(!autoRun && model.isThreadRunning()) {
		model.stopThread();
	}
	else if(autoRun && !model.isThreadRunning()) {
		model.startThread();
	}

	// write fbo to ofImage
	fbo.readToPixels(imgIn.getPixels());

	// async update on model input
	if(model.readyForInput()) {

		// read tensor from ofImage
		input = ofxTF2::imageToTensor(imgIn);

		// feed input into model
		model.update(input);

		// end measurment
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		ofLog() << "Run took: " << diff.count() << " s or ~"
		        << (int)(1.0/diff.count()) << " fps";
	}

	// async read from model output
	if(model.isOutputNew()) {

		// pull output from model
		output = model.getOutput();

		// write tensor to ofImage
		ofxTF2::tensorToImage(output, imgOut);
		imgOut.update();

		// start new measurement
		start = std::chrono::system_clock::now();
	}
}

//--------------------------------------------------------------
void ofApp::draw() {

	// DISPLAY STUFF
	std::stringstream str;
	str << "ENTER : Toggle auto run " << (autoRun ? "(X)" : "( )") << std::endl;
	str << "DEL   : Clear drawing " << std::endl;
	str << "d     : Toggle draw mode " << (drawMode == 0 ? "(draw)" : "(boxes)") << std::endl;
	str << "c/v   : Change draw radius (" << drawRadius << ")" << std::endl;
	str << "z/x   : Change draw color " << std::endl;
	str << "i     : Get color from mouse" << std::endl;
	str << std::endl;
	str << "Draw in the box on the left" << std::endl;
	str << "or drag an image (PNG) into it" << std::endl;

	ofPushMatrix();
		if(!drawImage(fbo, "fbo (draw in here)")) {
			str << "fbo not allocated !!" << std::endl;
		}
		// just to check fbo is reading correctly
		//if(!drawImage(imgIn, "imgIn")) {
		//	str << "imgIn not allocated !!" << srd::endl;
		//}
		if(!drawImage(imgOut, "imgOut")) {
			str << "imgOut not allocated !!" << std::endl;
		}
	ofPopMatrix();

	// draw info texts
	ofSetColor(150);
	ofDrawBitmapString(str.str(), fbo.getWidth() + 10, fbo.getHeight() + 40);
	ofDrawBitmapString(ofToString((int)ofGetFrameRate()) + " fps", ofGetWidth() - 54, 12);

	// draw colors
	ofFill();
	int x = 0;
	int y = fbo.getHeight() + 30;

	// draw current color
	ofSetColor(drawColor);
	ofDrawCircle(x + paletteDrawSize/2, y + paletteDrawSize/2, paletteDrawSize/2);
	ofSetColor(200);
	ofDrawBitmapString("Current draw color\n(change with z/x keys)",
	                   x + paletteDrawSize + 10, y + paletteDrawSize/2);
	y += paletteDrawSize + 10;

	// draw color palette
	for(int i = 0; i < colors.size(); i++) {
		ofSetColor(colors[i]);
		ofDrawCircle(x + paletteDrawSize/2, y + paletteDrawSize/2, paletteDrawSize/2);

		// draw outline if selected color
		if(colors[i] == drawColor) {
			ofPushStyle();
			ofNoFill();
			ofSetColor(255);
			ofSetLineWidth(3);
			ofDrawRectangle(x, y, paletteDrawSize, paletteDrawSize);
			ofPopStyle();
		}

		x += paletteDrawSize;

		// wrap around if doesn't fit on screen
		if(x > ofGetWidth() - paletteDrawSize) {
			x = 0;
			y += paletteDrawSize;
		}
	}

	// display drawing helpers
	ofNoFill();
	switch(drawMode) {
		case 0: // draw
			ofSetLineWidth(3);
			ofSetColor(ofColor::black);
			ofDrawCircle(ofGetMouseX(), ofGetMouseY(), drawRadius + 1);

			ofSetLineWidth(3);
			ofSetColor(drawColor);
			ofDrawCircle(ofGetMouseX(), ofGetMouseY(), drawRadius);

			break;
		case 1: // draw boxes
			if(ofGetMousePressed(0)) {
				ofSetLineWidth(3);
				ofSetColor(ofColor::black);
				ofDrawRectangle(mousePressPos.x - 1, mousePressPos.y - 1,
				                ofGetMouseX() - mousePressPos.x + 3,
				                ofGetMouseY() - mousePressPos.y + 3);

				ofSetLineWidth(3);
				ofSetColor(drawColor);
				ofDrawRectangle(mousePressPos.x, mousePressPos.y,
				                ofGetMouseX() - mousePressPos.x,
				                ofGetMouseY() - mousePressPos.y);
			}
			break;
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch(key) {

		case 'd':
		case 'D':
			drawMode = 1 - drawMode;
			break;

		case 'c':
			if(drawRadius > 0) {
				drawRadius--;
			}
			break;

		case 'v':
			drawRadius++;
			break;

		case 'z':
			drawColorIndex--;
			if(drawColorIndex < 0) {
				drawColorIndex += colors.size(); // wrap around
			}
			drawColor = colors[drawColorIndex];
			break;

		case 'x':
			drawColorIndex++;
			if(drawColorIndex >= colors.size()) {
				drawColorIndex -= colors.size(); // wrap around
			}
			drawColor = colors[drawColorIndex];
			break;

		case 'i':
		case 'I':
			if(ofGetMouseX() < fbo.getWidth() && ofGetMouseY() < fbo.getHeight()) {
				drawColor = imgIn.getColor(ofGetMouseX(), ofGetMouseY());
			}
			break;

		case OF_KEY_DEL:
		case OF_KEY_BACKSPACE:
			fbo.begin();
			ofClear(255);
			fbo.end();
			break;

		case OF_KEY_RETURN:
			autoRun ^= true;
			break;
	}
}

//--------------------------------------------------------------
void ofApp::mouseDragged( int x, int y, int button) {
	switch(drawMode) {
		case 0: // draw
			fbo.begin();
			ofSetColor(drawColor);
			ofFill();
			if(drawRadius > 0) {
				ofDrawCircle(x, y, drawRadius);
				ofSetLineWidth(drawRadius * 2);
			}
			else {
				ofSetLineWidth(0.1f);
			}
			ofDrawLine(x, y, ofGetPreviousMouseX(), ofGetPreviousMouseY());
			fbo.end();
			break;
		case 1: // draw boxes
			break;
	}
}

//--------------------------------------------------------------
void ofApp::mousePressed( int x, int y, int button) {
	mousePressPos = glm::vec2(x, y);
	mouseDragged(x, y, button);
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {
	switch(drawMode) {
		case 0: // draw
			break;
		case 1: // draw boxes
			fbo.begin();
			ofSetColor(drawColor);
			ofFill();
			ofDrawRectangle(mousePressPos.x, mousePressPos.y,
			                x - mousePressPos.x, y - mousePressPos.y);
			fbo.end();
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
	if(dragInfo.files.empty()) {
		return;
	}
	std::string file_path = dragInfo.files[0];

	// only PNGs work for some reason when Tensorflow is linked in
	ofImage img;
	img.load(file_path);
	if(img.isAllocated()) {
		fbo.begin();
		ofSetColor(255);
		ofFill();
		ofDrawRectangle(0, 0, fbo.getWidth(), fbo.getHeight()); // clear
		img.draw(0, 0, fbo.getWidth(), fbo.getHeight());
		fbo.end();
	}
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

// PRIVATE

//--------------------------------------------------------------
// setup the drawing tool by folder name
void ofApp::setupDrawingTool(std::string model_dir) {

	model_dir = ofToDataPath(model_dir);

	// load test image
	ofLogVerbose() << "Loading test image";
	ofImage img;
	img.load(ofFilePath::join(model_dir, "shoe.png"));
	if(img.isAllocated()) {
		fbo.begin();
		ofSetColor(255);
		img.draw(0, 0, fbo.getWidth(), fbo.getHeight());
		fbo.end();
	}
	else {
		ofLogError() << "Test image not found";
	}

	// load color palette for drawing
	ofLogVerbose() << "Loading color palette";
	colors.clear();
	ofBuffer buf;
	buf = ofBufferFromFile(ofFilePath::join(model_dir, "/palette.txt"));
	if(buf.size() > 0) {
		for(const auto& line : buf.getLines()) {
			ofLogVerbose() << line;
			if(line.size() == 6) { // if valid hex code
				colors.push_back(ofColor::fromHex(ofHexToInt(line)));
			}
		}
		drawColorIndex = 0;
		if(colors.size() > 0) {
			drawColor = colors[0];
		}
	}
	else {
		ofLogError() << "Palette info not found";
	}

	// load default brush info
	ofLogVerbose() << "Loading default brush info";
	buf = ofBufferFromFile(ofFilePath::join(model_dir, "/default_brush.txt"));
	if(buf.size() > 0) {
		auto str_info = buf.getLines().begin().asString();
		ofLogVerbose() << str_info;
		auto str_infos = ofSplitString(str_info, " ", true, true);
		if(str_infos[0] == "draw") {
			drawMode = 0;
		}
		else if(str_infos[0] == "box") {
			drawMode = 1;
		}
		else {
			ofLogError() << "Unknown draw mode: " << str_infos[0];
		}
		drawRadius = ofToInt(str_infos[1]);
	}
	else {
		ofLogError() << "Default brush info not found";
	}
}

//--------------------------------------------------------------
// draw image or fbo etc with border and label
// typename T must have draw(x,y), isAllocated(), getWidth(), getHeight()
template <typename T>
bool ofApp::drawImage(const T& img, string label) {
	if(img.isAllocated()) {
		ofSetColor(255);
		ofFill();

		// draw image
		img.draw(0, 0);

		// draw border
		ofNoFill();
		ofSetColor(200);
		ofSetLineWidth(1);
		ofDrawRectangle(0, 0, img.getWidth(), img.getHeight());

		// draw label
		ofDrawBitmapString(label, 10, img.getHeight() + 15);

		// next position
		ofTranslate(img.getWidth(), 0);

		return true;
	}
	return false;
}
