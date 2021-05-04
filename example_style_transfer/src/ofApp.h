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

// uncomment this to use a live camera otherwise, we'll use an image file
#define USE_LIVE_VIDEO

// custom ofxTF2::ThreadedModel implementation with custom pre- & postprocessing
class ImageToImageModel : public ofxTF2::ThreadedModel {

	public:

		// override the runModel function of ThreadedModel
		// this way the thread will take this augmented function
		// otherwise it would call runModel with no way of pre-/postprocessing
		cppflow::tensor runModel(const cppflow::tensor & input) const override {
			// cast data type and expand to batch size of 1
			auto inputCast = cppflow::cast(input, TF_UINT8, TF_FLOAT);
			inputCast = cppflow::expand_dims(inputCast, 0);
			// call to super
			auto output = Model::runModel(inputCast);
			// postprocess: last layer = (tf.nn.tanh(x) * 150 + 255. / 2)
			return ofxTF2::mapTensorValues(output, -22.5f, 277.5f, 0.0f, 255.0f);
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

		std::vector<std::string> modelPaths; // paths to available models
		std::size_t modelIndex = 0; // current model path index
		std::string modelName = ""; // current model path name
		bool newInput = false;      // is there new input to process?
		bool autoLoad = true;       // load models automatically?
		float loadTimestamp = 0;    // last model load time stamp
		float loadTimeSeconds = 10; // how long to wait before loading models
		
		ImageToImageModel model;
		cppflow::tensor input;
		cppflow::tensor output;
		int nnWidth = 640;
		int nnHeight = 480;

	#ifdef USE_LIVE_VIDEO
		ofVideoGrabber vidIn;
		int camWidth = 640;
		int camHeight = 480;
	#else
		ofImage imgIn;
	#endif
		ofImage imgOut;
};
