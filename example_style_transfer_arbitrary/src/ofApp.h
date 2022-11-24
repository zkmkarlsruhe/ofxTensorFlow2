/*
 * Example made with love by Jonathan Frank 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */
#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "ofxStyleTransfer.h"

// uncomment this to use a live camera, otherwise we'll use a video file
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

		/// goto prev style in the stylePaths vector
		void prevStyle();

		/// goto next style in the stylePaths vector
		void nextStyle();

		/// set style from given input image
		void setStyle(std::string & path);

		ofxStyleTransfer styleTransfer; ///< model wrapper
		ofFloatImage imgOut; ///< output image

		// video source
		#ifdef USE_LIVE_VIDEO
			ofVideoGrabber video;
			bool mirror = true;
		#else
			ofVideoPlayer video;
		#endif

		// image input & output size
		const static int imageWidth = 640;
		const static int imageHeight = 480;

		// paths to available style images
		std::vector<std::string> stylePaths = {
			"style/wave.jpg",
			"style/ZKM000092996.jpg",
			"style/chipset.jpg",
			"style/mondrian.jpg"
		};
		std::size_t styleIndex = 0; ///< current model path index
};
