#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "ofxMovenet.h"

#define USE_VIDEO



class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);

		ofVideoPlayer video;
		ofxMovenet movenet;

		
};
