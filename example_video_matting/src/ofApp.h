/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"

// uncomment this to use a live camera otherwise, we'll use a video file
//#define USE_LIVE_VIDEO

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

		// neural net input size
		float nnWidth = 1920;
		float nnHeight = 1080;

		// input
		#ifdef USE_LIVE_VIDEO
			int camWidth = 1920;
			int camHeight = 1080;
			ofVideoGrabber video;
		#else
			ofVideoPlayer video;
		#endif

		// model
		ofxTF2::Model model;

		std::vector<cppflow::tensor> inputs;

		ofImage imgMask;
		ofImage imgBackground;
		ofImage imgOut;
};
