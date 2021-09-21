#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"


class depthModel : public ofxTF2::ThreadedModel {

	public:

		cppflow::tensor runModel(const cppflow::tensor & input) const override {
			auto inputCast = cppflow::cast(input, TF_UINT8, TF_FLOAT);
			inputCast = inputCast / 255.f;
			inputCast = cppflow::expand_dims(inputCast, 0);

			auto output = Model::runModel(inputCast);
			
			return output;
			
		}
};
class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		

		ofVideoPlayer video;
		ofxTF2::Model model;

		ofImage bg;
		ofImage mask;
		ofImage outputMasked;

		std::vector<cppflow::tensor> vectorOfInputTensors;
};
