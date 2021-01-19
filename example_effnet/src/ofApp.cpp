#include "ofApp.h"
#include "cppflow/cppflow.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_effnet");

	// use TensorFlow ops through the cppflow wrappers
	// load a jpeg picture cast it to float and add a dimension for batches
	auto input = cppflow::decode_jpeg(cppflow::read_file(ofToDataPath("my_cat.jpg")));
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::expand_dims(input, 0);

	// load and infer the model
	cppflow::model model(ofToDataPath("model"));
	auto output = model(input);

	// interpret the output
	auto maxLabel = cppflow::arg_max(output, 1);
	ofLog() << "Maximum likelihood: " << maxLabel;

	// access each element via the internal vector
	auto outputVector = output.get_data<float>();
	
	ofLog() << "[281] tabby cat: " << outputVector[281];
	ofLog() << "[282] tiger cat: " << outputVector[282];
	ofLog() << "[283] persian cat: " << outputVector[283];
	ofLog() << "[284] siamese cat: " << outputVector[284];
	ofLog() << "[285] egyptian cat: " << outputVector[285];
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
