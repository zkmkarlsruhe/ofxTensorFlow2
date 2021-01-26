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

//#define USE_LIVE_VIDEO // uncomment this to use a live camera
						 // otherwise, we'll use an image file

// TODO: add "draw a shoe" canvas for live input?
// FIXME: does live camera input make sense for the model?
class ofApp : public ofBaseApp{

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

		// unless your computer is fast, use the threaded model so as not to
		// block the GUI while the model is processing
		ofxTF2ThreadedModel model;
		cppflow::tensor input;
		cppflow::tensor output;
		int nnWidth;
		int nnHeight;

		#ifdef USE_LIVE_VIDEO
			ofVideoGrabber vidSrc;
			int camWidth;
			int camHeight;
		#else
			ofImage imgSrc;
		#endif
		ofImage imgIn;
		ofImage imgOut;

		// time metrics
		std::chrono::time_point<std::chrono::system_clock> start;
		std::chrono::time_point<std::chrono::system_clock> end;
};
