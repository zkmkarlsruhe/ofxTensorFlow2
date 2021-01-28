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
		static constexpr std::size_t downsamplingFactor = 3;
		static constexpr std::size_t bufferSize = 1024;
		static constexpr std::size_t bufferSizeDownsampled = bufferSize / downsamplingFactor;
		static constexpr std::size_t samplingRate = 48000;
		
		// this buffer keeps the previous audio buffer since volume detection has some latency 
		std::vector<float> previousBuffer;
		
		// volume
		std::vector<float> volHistory;
		float smoothedVol;
		float scaledVol;
		float volThreshold;

		std::vector<float> sample;
		std::string displayLabel;

		// neural network
		cppflow::model *model = nullptr;
		cppflow::tensor output;
		static constexpr std::size_t inputSeconds = 1;
		static constexpr std::size_t inputSamplingRate = 16000;
		static constexpr std::size_t inputSize = inputSamplingRate * inputSeconds;

		// neural network control logic
		bool trigger;
		bool enable;
		bool recording;
		int recordingCounter;
		static constexpr std::size_t recordingCounterMax = samplingRate / bufferSize;
		static constexpr float minConfidence = 0.75;
};
