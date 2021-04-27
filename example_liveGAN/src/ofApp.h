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

// custom ofxTF2::ThreadedModel implementation with pre- & postprocessing
class SimpleModel : public ofxTF2::ThreadedModel {
	public:
		virtual cppflow::tensor runModel(const cppflow::tensor & input) const override {
			auto output_img = Model::runModel(input);
			return ofxTF2::mapTensorValues(output_img, -1, 1, 0, 255);
		}
};

class ofApp : public ofBaseApp {

	public:
		void setup();
		void update();
		void draw();
		
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y);
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		void audioIn(ofSoundBuffer & input);
		
		// neural network
		SimpleModel generator;

		// audio 
		ofSoundStream soundStream;
		std::vector<float> audio;
		bool newAudio = false;
		bool newSpec = false;

		// spectrum
		std::vector<float> curSpec;
		std::vector<float> spec;
		int64_t latentSize;

		// output image
		cppflow::tensor output;
		int nnWidth = 256;
		int nnHeight = 256;
		ofImage imgOut;
};
