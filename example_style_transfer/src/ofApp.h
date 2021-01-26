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

#define USE_LIVE_VIDEO // uncomment this to use a live camera
					// otherwise, we'll use an image file

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

		bool loadNewModel;
		std::size_t modelCounter;
		std::size_t frameCounter;
		std::size_t waitNumCamFrames;

		std::vector<std::string> modelPaths;
		
		ofxTF2ThreadedModel model;
		cppflow::tensor input;
		cppflow::tensor output;
		int nnWidth;
		int nnHeight;

		#ifdef USE_LIVE_VIDEO
			ofVideoGrabber vidIn;
			int camWidth;
			int camHeight;
		#endif
		ofImage imgIn;
		ofImage imgOut;
};
