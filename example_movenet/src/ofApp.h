/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#pragma once

#include "ofMain.h"
#include "ofxMovenet.h"

// uncomment this to use a live camera, otherwise we'll use a video file
//#define USE_LIVE_VIDEO

class ofApp : public ofBaseApp {

	public:
		void setup();
		void update();
		void draw();
		void exit();

		void keyPressed(int key);

		// neural net input size
		std::size_t nnWidth = 512;
		std::size_t nnHeight = 288;

		// input
		#ifdef USE_LIVE_VIDEO
			ofVideoGrabber video;
			int camWidth = 640;
			int camHeight = 480;
			ofImage imgOut;
			bool mirror = true;
		#else
			ofVideoPlayer video;
		#endif

		// model
		ofxMovenet movenet;
};
