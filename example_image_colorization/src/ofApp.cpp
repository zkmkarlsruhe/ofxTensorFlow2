#include "ofApp.h"


//--------------------------------------------------------------
void imageToLAB(ofFloatImage & imgIn){
	cv::Mat imgMat = ofxCv::toCv(imgIn);
	if (imgIn.getImageType() == OF_IMAGE_GRAYSCALE) {
		cv::cvtColor(imgMat, imgMat, CV_GRAY2RGB);
	}
	cv::cvtColor(imgMat, imgMat, CV_RGB2Lab);
}

//--------------------------------------------------------------
void LABtoImage(ofFloatImage & imgIn){
	cv::Mat imgMat = ofxCv::toCv(imgIn);
	cv::cvtColor(imgMat, imgMat, CV_Lab2RGB);
}

//--------------------------------------------------------------
cppflow::tensor runInference(cppflow::tensor & input, const ofxTF2::Model & model, int width, int height) {
	cppflow::tensor inputResized;
	cppflow::tensor output;
	std::vector<cppflow::tensor> vectorOfInputTensors;
	// expand and resize the input
	input = cppflow::expand_dims(input, 0);
	inputResized = cppflow::resize_bicubic(input, cppflow::tensor({ 256, 256 }), true);
	// compute, scale and resize the remaining channels
	output = model.runModel(inputResized);
	output = cppflow::mul(output, cppflow::tensor({ 128.f }));
	output = cppflow::resize_bicubic(output, cppflow::tensor({ height, width }), true);
	// concat the output
	vectorOfInputTensors = { input, output };
	return cppflow::concat(cppflow::tensor({ 3 }), vectorOfInputTensors);
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_image_colorization");

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
	videoPlayer.load("sunset_baw.mp4");
	width = videoPlayer.getWidth();
	height = videoPlayer.getHeight();
	imgOut.allocate(width, height, OF_IMAGE_COLOR);
	imgIn.allocate(width, height, OF_IMAGE_COLOR);
	videoPlayer.play();
#else
	// load image and allocate memory
	imgIn.load("wald.jpg");
	width = imgIn.getWidth();
	height = imgIn.getHeight();
	imgOut.allocate(width, height, OF_IMAGE_COLOR);
	// convert to LAB
	imageToLAB(imgIn);
	// convert the image to pixels
	input = ofxTF2::pixelsToTensor(imgIn.getPixels().getChannel(0));
	// compute the colorized image
	output = runInference(input, model, width, height);
	// convert image
	ofxTF2::tensorToImage(output, imgOut);
	LABtoImage(imgOut);
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
		imgIn.setFromPixels(videoPlayer.getPixels()); // convert to ofFloatImage
		imgIn.update();
		imageToLAB(imgIn); // convert to LAB space
		input = ofxTF2::pixelsToTensor(imgIn.getPixels().getChannel(0)); // convert first channel to tensor
		// compute the colorized image
		output = runInference(input, model, width, height);
		// convert image
		ofxTF2::tensorToImage(output, imgOut);
		LABtoImage(imgOut);
		imgOut.update();
	}
#endif
}

//--------------------------------------------------------------
void ofApp::draw() {
	imgIn.draw(20, 20, 480, 360);
	imgOut.draw(500, 20, 480, 360);
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
