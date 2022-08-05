/*
 * Example made with love by Jonathhhan 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */
#include "ofApp.h"

//--------------------------------------------------------------
std::vector<std::pair<std::vector<float>, int>>
computeBoundingBoxes(cppflow::tensor & input, const ofxTF2::Model & model,
	                 std::vector<std::pair<int, float>> & id) {
	cppflow::tensor output;
	std::vector<float> vec;
	std::vector<float>::const_iterator first;
	std::vector<float>::const_iterator last;
	cppflow::tensor inputResized;

	// expand, cast, scale and resize the input image
	input = cppflow::expand_dims(input, 0);
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::mul(input, cppflow::tensor({1/255.f}));
	inputResized = cppflow::resize_bicubic(input, cppflow::tensor({416, 416}), true);

	// run the model on the input
	output = model.runModel(inputResized);

	// compute the bounding boxes and add them to the id array
	ofxTF2::tensorToVector(output, vec);
	std::vector<std::vector<float>> boundings;
	for(int i = 0; i < vec.size() / 84; i++) {
		first = vec.begin() + 84. * i;
		last = vec.begin() + 84. * i + 4;;
		std::vector<float> newVec(first, last);
		boundings.push_back(newVec);
		first = vec.begin() + 84. * i + 4;
		last = vec.begin() + 84. * i + 84;
		vector<float> newVecId(first, last);
		int maxElementIndex = max_element(newVecId.begin(), newVecId.end()) - newVecId.begin();
		float maxElement = *max_element(newVecId.begin(), newVecId.end());
		id.push_back(std::make_pair(maxElementIndex, maxElement));
	}
	return nms(boundings, 0.9);
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_yolo_v4");
	ofNoFill();

	// ofxTF2 setup
	if(!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}
	if(!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({"serving_default_input_1"}, {"StatefulPartitionedCall"});

	ofBuffer buffer = ofBufferFromFile("cocoClasses.txt");
	for(auto& line : buffer.getLines()) {
		cocoClasses.push_back(line);
	}

#ifdef USE_MOVIE
	videoPlayer.load("movie.mp4");
	videoPlayer.play();
#else
	imgIn.load("image.jpg");
	input = ofxTF2::imageToTensor(imgIn);
	rectangles = computeBoundingBoxes(input, model, id);
#endif
}

//--------------------------------------------------------------
void ofApp::update() {
#ifdef USE_MOVIE
	videoPlayer.update();
	if(videoPlayer.isFrameNew()) {
		id.clear();
		input = ofxTF2::pixelsToTensor(videoPlayer.getPixels());
		rectangles = computeBoundingBoxes(input, model, id);
	}
#endif
}

//--------------------------------------------------------------
void ofApp::draw() {
	ofSetColor(255);
#ifdef USE_MOVIE
	videoPlayer.draw(20, 20, 480, 360);
#else
	imgIn.draw(20, 20, 480, 360);
#endif
	ofSetColor(255, 0, 0);
	for(int i = 0; i < rectangles.size(); i++) {
		if(id[rectangles[i].second].second > 0.2) {
			ofDrawRectangle(rectangles[i].first[1] * 480 + 20,
				rectangles[i].first[0] * 360 + 20,
				rectangles[i].first[3] * 480 - rectangles[i].first[1] * 480,
				rectangles[i].first[2] * 360 - rectangles[i].first[0] * 360);
			ofDrawBitmapStringHighlight(
                "id: " + cocoClasses[id[rectangles[i].second].first] + ", prob: " +
                ofToString(id[rectangles[i].second].second), rectangles[i].first[1] * 480 + 30,
                                        rectangles[i].first[0] * 360 + 40);
		}
	}
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
