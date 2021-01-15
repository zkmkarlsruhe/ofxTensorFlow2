
#pragma once

#include <vector>

#include "ofMain.h"

#include "cppflow/cppflow.h"
#include "labels.h"

class ofApp : public ofBaseApp{
	
	public:

		// audio 
		ofSoundStream soundStream;
		// for ease of use: we want to keep it a multiple of the downsampling factor
		// micSamplingRate vs. neuralNetwork inputSamplingRate (16000)
		const std::size_t downsamplingFactor = 3;
		const std::size_t bufferSizeDownsampled = 200;
		const std::size_t bufferSize = bufferSizeDownsampled * downsamplingFactor;
		const std::size_t samplingRate = 48000;

		// volume
		std::vector<float> volHistory;
		float smoothedVol;
		float scaledVol;
		float volThreshold;

		std::vector<float> sample;

		// neural network
		cppflow::model model;
		cppflow::tensor output;
		static constexpr std::size_t inputSeconds = 1;
		static constexpr std::size_t inputSamplingRate = 16000;
		static constexpr std::size_t inputSize = inputSamplingRate * inputSeconds;

		// neural network control logic
		bool trigger;
		bool enable;
		bool recording;
		int recordingCounter;
		const std::size_t recordingCounterMax = samplingRate / bufferSize;
		
    //--------------------------------------------------------------
    ofApp(std::string modelPath)
    : sample(inputSize), model(ofToDataPath(modelPath))
      {
		  std::cout << "done loading neural network" << std::endl;
		  std::cout.flush();
	  }

	//--------------------------------------------------------------
	void setup(){ 
		
		ofSetVerticalSync(true);5
		ofSetCircleResolution(80);
		ofBackground(54, 54, 54);	
		
		volHistory.assign(400, 0.0);
		
		smoothedVol     = 0.0;
		scaledVol		= 0.0;
		volThreshold 	= 25;

		recordingCounter = 0;
		enable = true;
		trigger = false;
		recording = false;

		// sound stream settings
		soundStream.printDeviceList();
		ofSoundStreamSettings settings;
		auto devices = soundStream.getMatchingDevices("default");
		if(!devices.empty()){
			settings.setInDevice(devices[0]);
		}
		settings.setInListener(this);
		settings.sampleRate = samplingRate;
		settings.numOutputChannels = 0;
		settings.numInputChannels = 1;
		settings.bufferSize = bufferSize;
		soundStream.setup(settings);

		// neural network warm up
		auto test = cppflow::fill({1, 16000}, 1.0f);
		output = model(test);
	}

	//--------------------------------------------------------------
	void update(){

		//lets scale the vol up to a 0-1 range 
		scaledVol = ofMap(smoothedVol, 0.0, 0.17, 0.0, 1.0, true);

		//lets record the volume into an array
		volHistory.push_back(scaledVol);
		
		//if we are bigger the the size we want to record - lets drop the oldest value
		if( volHistory.size() >= 400 ){
			volHistory.erase(volHistory.begin(), volHistory.begin()+1);
		}

		if (trigger) {

			// convert recorded sample to a single batch
			cppflow::tensor input(sample, {1, 16000});

			// inference
			auto output_vector = model(input).get_data<float>();

			// postprocessing
			auto maxElem = std::max_element(output_vector.begin(), output_vector.end());
			int argMax = std::distance(output_vector.begin(), maxElem);

			// label look up
			std::cout << "Label: " << labelsMap[argMax] << std::endl;
			std::cout.flush();
			
			// release the trigger signal
			trigger = false;
			enable = true;
		}
	}

	//--------------------------------------------------------------
	void draw(){
		
		ofSetColor(225);
		
		ofNoFill();
		
		// draw the average volume:
		ofPushStyle();
			ofPushMatrix();
			ofTranslate(565, 170, 0);
				
			ofSetColor(225);
			
			ofSetColor(245, 58, 135);
			ofFill();		
			ofDrawCircle(200, 200, scaledVol * 190.0f);
			
			//lets draw the volume history as a graph
			ofBeginShape();
			for (unsigned int i = 0; i < volHistory.size(); i++){
				if( i == 0 ) ofVertex(i, 400);

				ofVertex(i, 400 - volHistory[i] * 70);
				
				if( i == volHistory.size() -1 ) ofVertex(i, 400);
			}
			ofEndShape(false);		
				
			ofPopMatrix();
		ofPopStyle();
			
	}

	//--------------------------------------------------------------
	void audioIn(ofSoundBuffer & input){
		
		float curVol = 0.0;

		// calculate the root mean square which is a rough way to calculate volume
		for (size_t i = 0; i < input.getNumFrames(); i++){
			float vol = input[i]*0.5;
			curVol += vol * vol;
		}
		curVol /= (float)input.getNumFrames();
		curVol = sqrt( curVol );
		smoothedVol *= 0.93;
		smoothedVol += 0.07 * curVol;

		// trigger recording
		if (ofMap(curVol, 0.0, 0.17, 0.0, 1.0, true) * 100 >= volThreshold){
			if (enable){
				recording = true;
				enable = false;
				recordingCounter = 0;
				std::cout << "recording" << std::endl;
				std::cout.flush();
			}
		}

		// save and downsample a recordings
		// then trigger the neural network
		if (recording == true){

			// downsample by an integer
			int recOffset = recordingCounter * bufferSizeDownsampled;
			for (int i=0; i<bufferSizeDownsampled; i++) {
				float sum = 0.0; int offset = i*downsamplingFactor;
				for (int j=0; j<downsamplingFactor; j++){sum += input[offset+j];}
				sample[recOffset + i] = sum / downsamplingFactor;
			}

			recordingCounter++;
			if (recordingCounter >= recordingCounterMax){
				recording = false;
				trigger = true;
				std::cout << "trigger" << std::endl;
				std::cout.flush();
			}
		}
	}

	//--------------------------------------------------------------
	void keyPressed  (int key){ 

	}

	//--------------------------------------------------------------
	void keyReleased(int key){ 
		
	}

	//--------------------------------------------------------------
	void mouseMoved(int x, int y ){
		
	}

	//--------------------------------------------------------------
	void mouseDragged(int x, int y, int button){
		
	}

	//--------------------------------------------------------------
	void mousePressed(int x, int y, int button){
		
	}

	//--------------------------------------------------------------
	void mouseReleased(int x, int y, int button){

	}

	//--------------------------------------------------------------
	void mouseEntered(int x, int y){

	}

	//--------------------------------------------------------------
	void mouseExited(int x, int y){

	}

	//--------------------------------------------------------------
	void windowResized(int w, int h){

	}

	//--------------------------------------------------------------
	void gotMessage(ofMessage msg){

	}

	//--------------------------------------------------------------
	void dragEvent(ofDragInfo dragInfo){ 

	}

};
