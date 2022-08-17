/*
 * Example made with love by Jonathhhan 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */
#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"

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

	ofxTF2::Model model;
	cppflow::tensor style; // style image
	std::vector<cppflow::tensor> input; // {input image, style image}
	ofVideoPlayer videoPlayer; // source video
	ofFloatImage imgOut; // output image

	// image input & output size expected by model
	const static int imageWidth = 640;
	const static int imageHeight = 480;

	// style image size expected by model
	static const int styleWidth = 256;
	static const int styleHeight = 256;

	// paths to available style images
	std::vector<std::string> stylePaths = {
		"style/wave.jpg",
		"style/ZKM000092996.jpg",
		"style/chipset.jpg",
		"style/mondrian.jpg"
	};
	std::size_t styleIndex = 0; // current model path index
};
