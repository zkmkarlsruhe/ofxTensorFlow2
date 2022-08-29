/*
 * Example made with love by Jonathhhan 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */
#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "ofxYolo.h"

// uncomment this to use a video file, otherwise we'll use an image
#define USE_MOVIE

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

		// yolo model
		ofxYolo yolo;

		// input source
	#ifdef USE_MOVIE
		ofVideoPlayer video;
	#else
		ofImage imgIn;
	#endif
};
