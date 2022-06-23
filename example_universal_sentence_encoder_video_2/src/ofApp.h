#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "srtparser.h"

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

		ofVideoPlayer videoPlayer;
		ofxTF2::Model model;
		std::vector<SubtitleItem*> sub;
		std::vector<std::pair<cppflow::tensor, std::pair<int, int>>> vector_sub;
		std::vector<std::pair<cppflow::tensor, std::pair<int, int>>> vector_sub_copy;
		cppflow::tensor currentVector;
		int currentSubNo;
		int currentSubLenght;
		std::string show;
};
