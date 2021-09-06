#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "ofxMovenet.h"

#define USE_LIVE_VIDEO


class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);

		size_t nnWidth = 512;
		size_t nnHeight = 288;

        #ifdef USE_LIVE_VIDEO
            ofVideoGrabber video;
            int camWidth = 640;
            int camHeight = 480;
			ofImage imgOut;
        #else
			ofVideoPlayer video;
		#endif

		ofxMovenet movenet;

		
};
