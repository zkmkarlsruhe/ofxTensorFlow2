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
	ofSetWindowTitle("example_style_transfer_arbitrary");

	// ofxTF2 setup
	if(!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_90, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}

	// load model
	if(!styleTransfer.setup(imageWidth, imageHeight)) {
		std::exit(EXIT_FAILURE);
	}
	setStyle(stylePaths[styleIndex]);
	styleTransfer.startThread();

	// style image
	setStyle(stylePaths[styleIndex]);

	// video
	#ifdef USE_LIVE_VIDEO
		video.setDesiredFrameRate(30);
		video.setup(imageWidth, imageHeight);
	#else
		video.load("movie.mp4");
		video.setVolume(0); // blah blah blah
		video.play();
	#endif

	// output image
	imgOut.allocate(imageWidth, imageHeight, OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::update() {
	video.update();
	if(video.isFrameNew()) {
		// convert video frame to input tensor and resize as needed
		styleTransfer.setInput(video.getPixels());
	}
	if(styleTransfer.update()) {
		imgOut = styleTransfer.getOutput();
		#ifdef USE_LIVE_VIDEO
			if(mirror) {
				imgOut.mirror(false, true);
			}
		#endif
		imgOut.update();
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	imgOut.draw(20, 20, 480, 360);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch(key) {
		case OF_KEY_LEFT:
			prevStyle();
			break;
		case OF_KEY_RIGHT:
			nextStyle();
			break;
		case 'm':
			// toggle camera mirroring
			#ifdef USE_LIVE_VIDEO
				mirror = !mirror;
			#endif
			break;
		case ' ':
			// toggle video playback
			#ifndef USE_LIVE_VIDEO
				video.setPaused(!video.isPaused());
			#endif
			break;
		case 'r':
			// restart video
			#ifndef USE_LIVE_VIDEO
				video.stop();
				video.play();
			#endif
			break;
		default: break;
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

//--------------------------------------------------------------
void ofApp::prevStyle() {
	if(styleIndex == 0) {
		styleIndex = stylePaths.size()-1;
	}
	else {
		styleIndex--;
	}
	setStyle(stylePaths[styleIndex]);
}

//--------------------------------------------------------------
void ofApp::nextStyle() {
	styleIndex++;
	if(styleIndex >= stylePaths.size()) {
		styleIndex = 0;
	}
	setStyle(stylePaths[styleIndex]);
}

//--------------------------------------------------------------
void ofApp::setStyle(std::string & path) {
	ofImage styleImg;
	if(!styleImg.load(path)) {
		return;
	}
	styleTransfer.setStyle(styleImg.getPixels());
	ofLog() << "style: " << ofFilePath::getFileName(path);
}
