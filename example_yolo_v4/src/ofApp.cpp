#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofBuffer buffer = ofBufferFromFile("cocoClasses.txt");
	for (auto& line : buffer.getLines()) {
		cocoClasses.push_back(line);
	}
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_yolo_v4");
	ofNoFill();

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}

	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({ "serving_default_input_1" }, { "StatefulPartitionedCall" });

#ifdef USE_VIDEO
	videoPlayer.load("Frenzy.mp4");
	videoPlayer.play();
#else
	imgIn.load("eisenstein.jpg");
	input = ofxTF2::imageToTensor(imgIn);
	input = cppflow::expand_dims(input, 0);
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::div(input, cppflow::tensor({ 255.f }));
	input_resized = cppflow::resize_bicubic(input, cppflow::tensor({ 416, 416 }), true);

	output = model.runModel(input_resized);
	ofxTF2::tensorToVector(output, vec);
	std::vector<std::vector<float>> boundings;
	for (int i = 0; i < vec.size() / 84; i++) {
		first = vec.begin() + 84. * i;
		last = vec.begin() + 84. * i + 4;
		std::vector<float> newVec(first, last);
		boundings.push_back(newVec);
		first = vec.begin() + 84. * i + 4;
		last = vec.begin() + 84. * i + 84;
		vector<float> newVecId(first, last);
		int maxElementIndex = max_element(newVecId.begin(), newVecId.end()) - newVecId.begin();
		float maxElement = *max_element(newVecId.begin(), newVecId.end());
		id.push_back(std::make_pair(maxElementIndex, maxElement));
	}
	rectangles = nms(boundings, 0.9);
#endif
}

//--------------------------------------------------------------
void ofApp::update() {
#ifdef USE_VIDEO
	videoPlayer.update();
	if (videoPlayer.isFrameNew()) {
		id.clear();
		input = ofxTF2::pixelsToTensor(videoPlayer.getPixels());
		input = cppflow::expand_dims(input, 0);
		input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
		input = cppflow::div(input, cppflow::tensor({ 255.f }));
		input_resized = cppflow::resize_bicubic(input, cppflow::tensor({ 416, 416 }), true);

		output = model.runModel(input_resized);
		ofxTF2::tensorToVector(output, vec);
		std::vector<std::vector<float>> boundings;
		for (int i = 0; i < vec.size() / 84; i++) {
			first = vec.begin() + 84. * i;
			last = vec.begin() + 84. * i + 4;
			std::vector<float> newVec(first, last);
			boundings.push_back(newVec);
			first = vec.begin() + 84. * i + 4;
			last = vec.begin() + 84. * i + 84;
			std::vector<float> newVecId(first, last);
			int maxElementIndex = max_element(newVecId.begin(), newVecId.end()) - newVecId.begin();
			float maxElement = *max_element(newVecId.begin(), newVecId.end());
			id.push_back(std::make_pair(maxElementIndex, maxElement));
		}
		rectangles = nms(boundings, 0.9);
	}
#endif
}

//--------------------------------------------------------------
void ofApp::draw() {
	ofSetColor(255);
#ifdef USE_VIDEO
	videoPlayer.draw(20, 20, 480, 360);
#else
	imgIn.draw(20, 20, 480, 360);
#endif
	ofSetColor(255, 0, 0);
	for (int i = 0; i < rectangles.size(); i++) {
		if (id[rectangles[i].second].second > 0.2) {
			ofDrawRectangle(rectangles[i].first[1] * 480 + 20, rectangles[i].first[0] * 360 + 20, rectangles[i].first[3] * 480 - rectangles[i].first[1] * 480, rectangles[i].first[2] * 360 - rectangles[i].first[0] * 360);
			ofDrawBitmapStringHighlight("id: " + cocoClasses[id[rectangles[i].second].first] + ", prob: " + ofToString(id[rectangles[i].second].second), rectangles[i].first[1] * 480 + 30, rectangles[i].first[0] * 360 + 40);
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