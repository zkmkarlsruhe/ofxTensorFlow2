#include "ofApp.h"
#include "cppflow/ops.h"
#include "cppflow/model.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_basics");
	ofBackground(100, 100, 100);

	model = new cppflow::model(ofToDataPath("model"));

	// try to grab at this size
	camWidth = 640;
	camHeight = 480;

	nnWidth = 512;
	nnHeight = 512;

	videoGrabber.setDeviceID(0);
	videoGrabber.setDesiredFrameRate(30);
	videoGrabber.initGrabber(camWidth, camHeight);

	frameProcessed.allocate(nnWidth, nnHeight, OF_PIXELS_RGB);
	inputTexture.allocate(frameProcessed);
	outputTexture.allocate(frameProcessed);
}

//--------------------------------------------------------------
void ofApp::update(){

	videoGrabber.update();

	// check for new frame
	if(videoGrabber.isFrameNew()){

		// get the frame
		ofPixels & pixels = videoGrabber.getPixels();

		// resize pixels
		ofPixels resizedPixels(pixels);
		resizedPixels.resize(nnWidth, nnHeight);

		// copy to tensor
		cppflow::tensor input(
			  std::vector<float>(resizedPixels.begin(),
								  resizedPixels.end()),
								  {nnWidth, nnHeight, 3});

		// cast data type and expand to batch size of 1
		input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
		input = cppflow::expand_dims(input, 0);
		input = cppflow::div(input, cppflow::tensor({127.5f}));
		input = cppflow::add(input, cppflow::tensor({-1.0f}));

		// auto & input_vector = input.get_data<float>();
		// for(int i = 0; i < input_vector.size(); i++){
		// 	input_vector[i] = input_vector[i] / 255;
		// }

		// start neural network and time measurement
		auto start = std::chrono::system_clock::now();
		auto output = (*model)(input);
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end-start;

		// ofLog() << output;
		ofLog() << "Time: " << diff.count() << "s " << ofGetFrameRate() << " fps";

		// copy to output frame and postprocessing
		auto outputVector = output.get_data<float>();
		for(int i = 0; i < outputVector.size(); i++){
			frameProcessed[i] = (outputVector[i] + 1) * 127.5;
		}

		outputTexture.loadData(frameProcessed);
		inputTexture.loadData(resizedPixels);
	}
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofSetColor(255);
	videoGrabber.draw(20, 20);
	outputTexture.draw(20 + camWidth, 20, nnWidth, nnHeight);
	inputTexture.draw(20, 20 + camHeight, nnWidth, nnHeight);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	if(key == 's' || key == 'S'){
		videoGrabber.videoSettings();
	}
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
