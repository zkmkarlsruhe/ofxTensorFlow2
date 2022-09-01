/*
 * Example made with love by Jonathhhan 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */
#include "ofApp.h"

//--------------------------------------------------------------
// help to convert ofPixels to a float image tensor
template <typename T>
cppflow::tensor pixelsToFloatTensor(const ofPixels_<T> & pixels) {
	auto t = ofxTF2::pixelsToTensor(pixels);
	t = cppflow::expand_dims(t, 0);
	t = cppflow::cast(t, TF_UINT8, TF_FLOAT);
	t = cppflow::mul(t, cppflow::tensor({1.0f / 255.f}));
	return t;
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_style_transfer_arbitrary");

	// ofxTF2 setup
	if(!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_90, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}
	if(!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	std::vector<std::string> inputNames = {
		"serving_default_placeholder",
		"serving_default_placeholder_1"
	};
	std::vector<std::string> outputNames = {
		"StatefulPartitionedCall"
	};
	model.setup(inputNames, outputNames);

	// style image
	inputVector = {cppflow::tensor(0), cppflow::tensor(0)};
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
		cppflow::tensor image = pixelsToFloatTensor(video.getPixels());
		if(video.getHeight() != imageWidth || video.getWidth() != imageHeight) {
			image = cppflow::resize_bicubic(image, cppflow::tensor({imageHeight, imageWidth}), true);
		}
		inputVector[0] = image;

		// run model
		auto output = model.runMultiModel(inputVector);

		// convert output tensor to image
		ofxTF2::tensorToImage(output[0], imgOut);
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
	ofLog() << "style: " << ofFilePath::getFileName(path);

	// convert style image to float tensor
	style = pixelsToFloatTensor(styleImg.getPixels());

	// resize to expected model input size
	style = cppflow::resize_bicubic(style, cppflow::tensor({styleWidth, styleHeight}), true);
	inputVector[1] = style;
}
