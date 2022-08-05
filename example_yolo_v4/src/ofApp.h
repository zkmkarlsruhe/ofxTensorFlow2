/*
 * Example made with love by Jonathhhan 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */
#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "nms.hpp"

// uncomment this to use a video file otherwise, we'll use an image
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
	std::vector<std::string> cocoClasses;
	std::vector<std::pair<std::vector<float>, int>> rectangles;
	std::vector<std::pair<int, float>> id;

#ifdef USE_MOVIE
	ofVideoPlayer videoPlayer;
#else
	ofImage imgIn;
#endif

};
