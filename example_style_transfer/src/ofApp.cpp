/*
 * ofxTensorFlow2
 *
 * Copyright (c) 2021 ZKM | Hertz-Lab
 * Paul Bethge <bethge@zkm.de>
 * Dan Wilcox <dan.wilcox@zkm.de>
 *
 * BSD Simplified License.
 * For information on usage and redistribution, and for a DISCLAIMER OF ALL
 * WARRANTIES, see the file, "LICENSE.txt," in this distribution.
 *
 * This code has been developed at ZKM | Hertz-Lab as part of „The Intelligent 
 * Museum“ generously funded by the German Federal Cultural Foundation.
 */

#include "ofApp.h"
#include "ofLog.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_style_transfer");

	ofDirectory modelsDir(ofToDataPath("models"));
	modelsDir.listDir();
	
	// go through and print out all the paths
	for(int i = 0; i < modelsDir.size(); i++){
		ofDirectory sub(modelsDir.getPath(i));
		if (sub.isDirectory()){
			auto absSubPath = sub.getAbsolutePath();
   			ofLogNotice() << "Found model: " << absSubPath;
			modelPaths.push_back(absSubPath);
		}
	}

	model.load(modelPaths[0]);
	modelCounter = 1;
	frameCounter = 0;
	waitNumCamFrames = 100; 

	nnWidth = 1024;
	nnHeight = 1024;
	imgIn.allocate(nnWidth, nnWidth, OF_IMAGE_COLOR);
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);

#ifdef USE_LIVE_VIDEO
	// try to grab at this size
	camWidth = 640;
	camHeight = 480;
	vidIn.setDesiredFrameRate(30);
	vidIn.setup(camWidth, camHeight);
#endif
	model.startThread();
}

//--------------------------------------------------------------
void ofApp::update(){

	// load a new model from the directory every other waitNumCamFrames frames
	if (frameCounter >= waitNumCamFrames){
		frameCounter = 0;
		loadNewModel = true;
	}
	if (loadNewModel){
		loadNewModel = false;
		if (modelCounter >= modelPaths.size())
			modelCounter = 0;
		ofLogNotice() << "Load model: " << modelPaths[modelCounter];
		model.loadSafely(modelPaths[modelCounter]);
		modelCounter++;
	}

#ifdef USE_LIVE_VIDEO
	// create tensor from video
	vidIn.update();
	if(vidIn.isFrameNew()){	
		

		// get the frame
		ofPixels & pixels = vidIn.getPixels();

		// resize pixels
		ofPixels resizedPixels(pixels);
		resizedPixels.resize(nnWidth, nnHeight);

		// resizedPixels.resizeTo(imgIn);
		

		// copy to tensor
		input = cppflow::tensor(
			  std::vector<float>(resizedPixels.begin(),
								  resizedPixels.end()),
							  {nnHeight, nnWidth, 3});

#else
		std::string imgPath(ofToDataPath("cat512x512.jpg"));
		// std::string imgPath(ofToDataPath("cat640x480.jpg"));
		ofLog() << "Loading image: " << imgPath;
		// create tensor from image file
		input = cppflow::decode_jpeg(cppflow::read_file(imgPath));
		ofLog() << "Image loaded";
#endif

		if (model.readyForInput()){
			ofLog() << "Input is ready";
			// cast data type and expand to batch size of 1
			input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
			input = cppflow::expand_dims(input, 0);
			model.update(input);
		}


		// // start neural network and time measurement
		// auto start = std::chrono::system_clock::now();
		// // output = (*model)({{"serving_default_input_1", input}}, {"StatefulPartitionedCall"})[0];
		// auto end = std::chrono::system_clock::now();
		// std::chrono::duration<double> diff = end - start;
		// ofLog() << "Time: " << diff.count() << "s Fps: " << ofGetFrameRate();

		if (model.isOutputNew()){
			ofLog() << "Output is ready";
			auto output = model.getOutput();
			auto outputShape = output.shape().get_data<shape_t>();
			ofLog() << "outputShape: " << ofxTensorFlow2::vectorToString(outputShape);

			// copy output to image
			auto outputVector = output.get_data<float>();
			auto & outputPixels = imgOut.getPixels();
			for(int i = 0; i < outputPixels.size(); i++){
				outputPixels[i] = outputVector[i];
			}
			imgOut.update();

		}

		// copy input to image
		auto inputVector = input.get_data<float>();
		auto & inputPixels = imgIn.getPixels();
		for(int i = 0; i < inputPixels.size(); i++){
			inputPixels[i] = inputVector[i];
		}
		
		imgIn.update();

		frameCounter++;


#ifdef USE_LIVE_VIDEO
	}
	else{
		// try again later
		return;
	}
#endif
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofSetColor(255);
	imgOut.draw(0, 0);
	imgIn.draw(nnWidth, 0);
#ifdef USE_LIVE_VIDEO
	// vidIn.draw(nnWidth, 0, camWidth*2, camHeight*2);
#endif
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
#ifdef USE_LIVE_VIDEO
	if(key == 's' || key == 'S'){
		vidIn.videoSettings();
	}
#endif
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){

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
