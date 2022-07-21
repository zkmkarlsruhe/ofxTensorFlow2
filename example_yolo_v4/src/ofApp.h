#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "nms.hpp"

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

	ofxTF2::Model model;
	cppflow::tensor input;
	std::vector<string> cocoClasses;
	std::vector<std::pair<std::vector<float>, int>> rectangles;
	std::vector<std::pair<int, float>> id;

#ifdef USE_MOVIE
	ofVideoPlayer videoPlayer;
#else
	ofImage imgIn;
#endif

};