#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_video_and_image_colorization");

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}

	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}

	model.setup({ "serving_default_input_1" }, { "StatefulPartitionedCall" });

#ifdef USE_VIDEO
	videoPlayer.load("Godard.mp4");
	imgOut.allocate(videoPlayer.getWidth(), videoPlayer.getHeight(), OF_IMAGE_COLOR);
	width = videoPlayer.getWidth();
	height = videoPlayer.getHeight();
	videoPlayer.play();
#else
	imgIn.load("wald.jpg");
	imgOut.allocate(imgIn.getWidth(), imgIn.getHeight(), OF_IMAGE_COLOR);
	width = imgIn.getWidth();
	height = imgIn.getHeight();
	input = ofxTF2::pixelsToTensor(imgIn.getPixels().getChannel(0));
	input = cppflow::expand_dims(input, 0);
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::div(input, cppflow::tensor({ 255.f / 100.f }));
	input_resized = cppflow::resize_bicubic(input, cppflow::tensor({ 256, 256 }), true);

	output = model.runModel(input_resized);
	output = cppflow::div(output, cppflow::tensor({ 1.f / 255.f }));
	output = cppflow::resize_bicubic(output, cppflow::tensor({ height, width }), true);
	vectorOfInputTensors = { input, output };
	output = cppflow::concat(cppflow::tensor({ 3 }), vectorOfInputTensors);
	ofxTF2::tensorToImage(output, imgOut);
	imgMat = ofxCv::toCv(imgOut);
	cv::cvtColor(imgMat, imgMat, CV_Lab2RGB);
	imgOut.update();
	imgOut.save("wald_colorized.jpg");
#endif
}

//--------------------------------------------------------------
void ofApp::update() {
#ifdef USE_VIDEO
	videoPlayer.update();
	if (videoPlayer.isFrameNew()) {
		input = ofxTF2::pixelsToTensor(videoPlayer.getPixels().getChannel(0));
		input = cppflow::expand_dims(input, 0);
		input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
		input = cppflow::div(input, cppflow::tensor({ 255.f / 100.f }));
		input_resized = cppflow::resize_bicubic(input, cppflow::tensor({ 256, 256 }), true);

		output = model.runModel(input_resized);
		output = cppflow::cast(output, TF_UINT8, TF_FLOAT);
		output = cppflow::div(output, cppflow::tensor({ 1.f / 255.f }));
		output = cppflow::resize_bicubic(output, cppflow::tensor({ height, width }), true);
		vectorOfInputTensors = { input, output };
		output = cppflow::concat(cppflow::tensor({ 3 }), vectorOfInputTensors);	
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