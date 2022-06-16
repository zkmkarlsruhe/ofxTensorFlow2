#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	cppflow::tensor tensor;
	std::vector<double> vec;

	ofSetWindowTitle("example_universal_sentence_encoder");

	// use only a portion of the GPU memory & grow as needed
	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}

	// load the model, bail out on error
	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({ {"serving_default_inputs:0"} }, { {"StatefulPartitionedCall_1:0"} });

	SubtitleParserFactory* subParserFactory = new SubtitleParserFactory(ofToDataPath("stranger.srt"));
	SubtitleParser* parser = subParserFactory->getParser();
	std::vector<SubtitleItem*> sub = parser->getSubtitles();
	for (auto element : sub) {
		tensor = model.runModel(cppflow::reshape(cppflow::tensor(element->getDialogue()), { -1 }));
		ofxTF2::tensorToVector(tensor, vec);
		vector_sub.push_back(std::make_pair(vec, element->getDialogue()));
	}
	vector_sub_copy = vector_sub;
	std::cout << "Subtitles loaded." << std::endl;
	currentVector = vector_sub_copy[0].first;
	currentString = vector_sub_copy[0].second;
	vector_sub_copy.erase(vector_sub_copy.begin());
}

//--------------------------------------------------------------
void ofApp::update() {

}

//--------------------------------------------------------------
void ofApp::draw() {
	ofDrawBitmapStringHighlight("Press a key!", 50, 50);
	ofDrawBitmapString(show, 50,100);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	std::vector<double> cosine;
	for (int x = 0; x < vector_sub_copy.size(); x++ ) {
		double cosine_similarity = 0;
		for (int i = 0; i < vector_sub_copy[x].first.size(); i++) {
			cosine_similarity += currentVector[i] * vector_sub_copy[x].first[i];
		}
		cosine.push_back(cosine_similarity);
	}
	int maxElementIndex = std::max_element(cosine.begin(), cosine.end()) - cosine.begin();
	double maxElement = *std::max_element(cosine.begin(), cosine.end());
	show = "Subtitle: " + ofToString(maxElementIndex) + ".\n\nThe cosine similarity between\n'"  + currentString + "'\nand\n'" + vector_sub_copy[maxElementIndex].second + "'\nis: " + ofToString(maxElement) + ".\n\nSubtitles left: " + ofToString(vector_sub_copy.size() - 1) + ".";
	currentVector = vector_sub_copy[maxElementIndex].first;
	currentString = vector_sub_copy[maxElementIndex].second;
	vector_sub_copy.erase(vector_sub_copy.begin() + maxElementIndex);
	if (vector_sub_copy.size() < 1) {
		vector_sub_copy = vector_sub;
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
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}