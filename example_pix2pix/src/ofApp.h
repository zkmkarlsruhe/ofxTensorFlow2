#include "ofMain.h"
#include "ofxTensorFlow2.h"


class ImageToImageModel : public ofxTF2ThreadedModel {

	public:
	// override the runModel function of ofxTF2ThreadedModel
	// this way the thread will take this augmented function 
    cppflow::tensor runModel(const cppflow::tensor & input) const override {
		
		// cast data type and expand to batch size of 1
		auto tempInput = cppflow::cast(input, TF_UINT8, TF_FLOAT);
		tempInput = cppflow::expand_dims(tempInput, 0);

		// apply preprocessing as in python to change range to -1 to 1
		tempInput = cppflow::div(tempInput, cppflow::tensor({127.5f}));
		tempInput = cppflow::sub(tempInput, cppflow::tensor({1.0f}));

		// call to super 
		auto output = ofxTF2Model::runModel(tempInput);

		// postprocess to change range to -1 to 1 and copy output to image
		output = cppflow::add(output, cppflow::tensor({1.0f}));
		output = cppflow::mul(output, cppflow::tensor({127.5f}));
		return output;
	}
};

//--------------------------------------------------------------
class ofApp : public ofBaseApp {
	public:

		// implemented standard functions
		void setup();
		void update();
		void draw();
		void keyPressed(int key);
		void mouseDragged( int x, int y, int button);
		void mousePressed( int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void dragEvent(ofDragInfo dragInfo);

		// not implemented
		void keyReleased(int key);
		void mouseMoved(int x, int y);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void gotMessage(ofMessage msg);

	private:

		void setupDrawingTool(string modelDir);

		template <typename T>
		bool drawImage(const T& img, string label);

		// Model
		ImageToImageModel model;
		cppflow::tensor input;
		cppflow::tensor output;
		int nnWidth;
		int nnHeight;

		// Draw
		ofFbo fbo;
		ofImage imgIn;
		ofImage imgOut;

		// model file management
		ofDirectory models_dir; // folder which contains drawing tool settings

		// color management for drawing
		vector<ofColor> colors; // contains color palette to be used for drawing
		int paletteDrawSize;
		int drawColorIndex;
		ofColor drawColor;

		// other vars
		bool autoRun;   // auto run every frame
		int drawMode;     	// draw vs boxes
		int drawRadius;
		ofVec2f mousePressPos;

		// time metrics
		std::chrono::time_point<std::chrono::system_clock> start;
		std::chrono::time_point<std::chrono::system_clock> end;
	};
