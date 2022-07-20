#include "ofApp.h"

//--------------------------------------------------------------
cppflow::tensor runInference(cppflow::tensor & input, const ofxTF2::Model & model, int width, int height) {
	cppflow::tensor inputResized;
	cppflow::tensor output;
	std::vector<cppflow::tensor> vectorOfInputTensors;
	// convert, scale and resize the input
	input = cppflow::expand_dims(input, 0);
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	inputResized = cppflow::mul(input, cppflow::tensor({ 100.f / 255.f }));
	inputResized = cppflow::resize_bicubic(inputResized, cppflow::tensor({ 256, 256 }), true);
	// compute, scale and resize the remaining channels
	output = model.runModel(inputResized);
	output = cppflow::mul(output, cppflow::tensor({ 255.f }));
	output = cppflow::resize_bicubic(output, cppflow::tensor({ height, width }), true);
	// concatenate the computed channels with the input
	vectorOfInputTensors = { input, output };
	return cppflow::concat(cppflow::tensor({ 3 }), vectorOfInputTensors);
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_video_and_image_colorization");

	// ofxTF2 setup
	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}
	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({ "serving_default_input_1" }, { "StatefulPartitionedCall" });

#ifdef USE_MOVIE
	// load video and allocate memory
	videoPlayer.load("movie2.mp4");
	width = videoPlayer.getWidth();
	height = videoPlayer.getHeight();
	imgOut.allocate(width, height, OF_IMAGE_COLOR);
	videoPlayer.play();
#else
	// load image and allocate memory
	imgIn.load("wald.jpg");
	width = imgIn.getWidth();
	height = imgIn.getHeight();
	imgOut.allocate(width, height, OF_IMAGE_COLOR);
	// convert the image to pixels
	input = ofxTF2::pixelsToTensor(imgIn.getPixels().getChannel(0));
	// compute the colorized image
	output = runInference(input, model, width, height);
	// convert image
	ofxTF2::tensorToImage(output, imgOut);
	imgMat = ofxCv::toCv(imgOut);
	cv::cvtColor(imgMat, imgMat, CV_Lab2RGB);
	imgOut.update();
	imgOut.save("wald_colorized.jpg");
#endif
}

//--------------------------------------------------------------
void ofApp::update() {
#ifdef USE_MOVIE
	videoPlayer.update();
	if (videoPlayer.isFrameNew()) {
		// get new frame
		input = ofxTF2::pixelsToTensor(videoPlayer.getPixels().getChannel(0));
		// compute the colorized image
		output = runInference(input, model, width, height);
		// convert image
		ofxTF2::tensorToImage(output, imgOut);
		imgMat = ofxCv::toCv(imgOut);
		cv::cvtColor(imgMat, imgMat, CV_Lab2RGB);
		imgOut.update();
	}
#endif
}

//--------------------------------------------------------------
void ofApp::draw() {
	imgOut.draw(20, 20, 480, 360);
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
