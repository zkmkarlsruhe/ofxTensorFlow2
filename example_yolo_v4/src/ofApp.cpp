/*
 * Example made with love by Jonathan Frank 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */
#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_yolo_v4");

	// set up yolo with path to model folder and txt file with COCO classes aka
	// identifiable object classification strings
	if(!yolo.setup("model", "classes.txt")) {
		std::exit(EXIT_FAILURE);
	}
	yolo.setNormalize(true); // normalize object bounding box coords?

	// input source
	#ifdef USE_MOVIE
		video.load("movie.mp4");
		video.play();
	#else
		imgIn.load("dog.jpg");
		yolo.setInput(imgIn.getPixels());
		yolo.update();
	#endif
}

//--------------------------------------------------------------
void ofApp::update() {
#ifdef USE_MOVIE
	video.update();
	if(video.isFrameNew()) {
		// feed input frame as pixels
		yolo.setInput(video.getPixels());
	}

	// run model on current input frame
	yolo.update();
#endif
}

//--------------------------------------------------------------
void ofApp::draw() {
	float x = 20, y = 20, w = 480, h = 360;
	ofSetColor(255);
#ifdef USE_MOVIE
	video.draw(x, y, w, h);
#else
	imgIn.draw(x, y, w, h);
#endif
	// draw detected objects
	if(yolo.getNormalize()) {
		// draw manually with normalized coords, requires yolo.setNormalize(true)
		ofNoFill();
		for(auto object : yolo.getObjects()) {
			if(object.index == 0) { // person
				ofSetColor(ofColor::blue);
			}
			else if(object.index == 2) { // car
				ofSetColor(ofColor::green);
			}
			else { // everything else...
				ofSetColor(ofColor::red);
			}
			ofDrawRectangle(object.bbox.x * w + x, object.bbox.y * h + y,
			                object.bbox.width * w, object.bbox.height * h);
			ofDrawBitmapStringHighlight(object.ident + "\n" + ofToString(object.confidence, 2),
			                            object.bbox.x * w + x, object.bbox.y * h + y);
		}
	}
	else {
		// draw within input image size
		ofPushMatrix();
		ofTranslate(x, y);
		ofScale(w / yolo.getWidth(), h / yolo.getHeight());
		yolo.draw();
		ofPopMatrix();
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch(key) {
		case ' ':
#ifdef USE_MOVIE
			// toggle video playback
			video.setPaused(!video.isPaused());
#endif
			break;
		default:
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
