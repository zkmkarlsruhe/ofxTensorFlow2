#pragma once

#include "ofMain.h"

#include "cppflow/cppflow.h"

// #define USE_LIVE_VIDEO // uncomment this to use a live camera
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

		cppflow::model *model = nullptr;
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
