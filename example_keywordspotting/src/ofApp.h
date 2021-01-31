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

#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "labels.h"

#include "utils.h"

class ofApp : public ofBaseApp{

	public:

		void setup();
		void update();
		void draw();

		void audioIn(ofSoundBuffer & input);

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		// audio 
		ofSoundStream soundStream;

		// for ease of use: we want to keep the buffersize a multiple of the downsampling factor
		// downsamplingFactor = micSamplingRate / neuralNetworkInputSamplingRate 
		std::size_t downsamplingFactor;
		std::size_t bufferSize;
		std::size_t samplingRate;
		
		// since volume detection has some latency be buffer past buffers
		AudioBufferFifo previousBuffers;
		std::size_t numPreviousBuffers;
		// sampleBuffers acts as a buffer for recording (could be fused)
		AudioBufferFifo sampleBuffers;
		std::size_t numBuffers;
		
		// volume
		float curVol;
		float smoothedVol;
		float scaledVol;
		float volThreshold;

		// display
		std::vector<float> volHistory;
		std::string displayLabel;

		// neural network	
		AudioClassifier model;
		cppflow::tensor output;
		std::size_t inputSeconds;
		std::size_t inputSamplingRate;
		std::size_t inputSize;

		// neural network control logic
		std::size_t recordingCounter;
		bool trigger;
		bool enable;
		bool recording;
		float minConfidence;
};
