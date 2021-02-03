#include "ofApp.h"


//--------------------------------------------------------------
void ofApp::setup() {
	ofSetColor(255);
	ofBackground(54, 54, 54);
	ofSetVerticalSync(true);
	ofSetLogLevel(OF_LOG_VERBOSE);
	ofSetFrameRate(60);

	ofSetWindowTitle("example_pix2pix");

	// neural network setup
	// the default model is edges2shoes and excepts [None, None, None, 3]
	model.load("model");
	nnWidth = 512;
	nnHeight = 512;
	
	// allocate fbo and images with correct dimensions, and no alpha channel
	ofLogVerbose() << "allocating fbo and images (" 
					<< nnWidth << ", " << nnHeight << ")";
	fbo.allocate(nnWidth, nnHeight, GL_RGB);
	imgIn.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
	imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);

	// color management for drawing
	paletteDrawSize = 50;
	drawColorIndex = 0;

	// other vars
	autoRun = true;    // auto run every frame
	drawMode = 0;     // draw vs boxes
	drawRadius = 10;

	// search for the drawing tool config files in bin/data/draw
	setupDrawingTool(ofToDataPath("draw"));

	// shorten idle time to have model check for input more frequently,
	// this may increase responsivity on faster machines but will use more cpu
	model.setIdleTime(10);

	// start the model background thread
	model.startThread();
}

void ofApp::update() {

	// start & stop the model
	if (!autoRun && model.isThreadRunning()){
		model.stopThread();
	}
	else if (autoRun && !model.isThreadRunning()){
		model.startThread();
	}

	// write fbo to ofImage
	fbo.readToPixels(imgIn.getPixels());

	// async update on model input
	if(model.readyForInput()) {

		// read tensor from ofImage
		input = ofxTF2::imageToTensor<float>(imgIn);

		// feed input into model
		model.update(input);

		// end measurment
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		ofLog() << "run took: " << diff.count() << " s or ~" << (int)(1.0/diff.count()) << " fps";
	}

	// async read from model output
	if(model.isOutputNew()) {

		// pull output from model
		output = model.getOutput();

		// write tensor to ofImage
		ofxTF2::tensorToImage<float>(output, imgOut);
		imgOut.update();

		// start new measurement
		start = std::chrono::system_clock::now();
	}
}


//--------------------------------------------------------------
void ofApp::draw() {

	// DISPLAY STUFF
	stringstream str;
	str << ofGetFrameRate() << endl;
	str << endl;
	str << "ENTER : toggle auto run " << (autoRun ? "(X)" : "( )") << endl;
	str << "DEL   : clear drawing " << endl;
	str << "d     : toggle draw mode " << (drawMode==0 ? "(draw)" : "(boxes)") << endl;
	str << "c/v   : change draw radius (" << drawRadius << ")" << endl;
	str << "z/x   : change draw color " << endl;
	str << "i     : get color from mouse" << endl;
	str << endl;
	str << "draw in the box on the left" << endl;
	str << "or drag an image (PNG) into it" << endl;
	str << endl;

	ofPushMatrix();
	{
		if(!drawImage(fbo, "fbo (draw in here)") ) str << "fbo not allocated !!" << endl;
		// if(!drawImage(imgIn, "imgIn") ) str << "imgIn not allocated !!" << endl; // just to check fbo is reading correctly
		if(!drawImage(imgOut, "imgOut") ) str << "imgOut not allocated !!" << endl;

		ofTranslate(20, 0);

		// draw texts
		ofSetColor(150);
		ofDrawBitmapString(str.str(), 0, 20);
	}
	ofPopMatrix();


	// draw colors
	ofFill();
	int x=0;
	int y=fbo.getHeight() + 30;

	// draw current color
	ofSetColor(drawColor);
	ofDrawCircle(x+paletteDrawSize/2, y+paletteDrawSize/2, paletteDrawSize/2);
	ofSetColor(200);
	ofDrawBitmapString("current draw color (change with z/x keys)", x+paletteDrawSize+10, y+paletteDrawSize/2);
	y += paletteDrawSize + 10;

	// draw color palette
	for(int i=0; i<colors.size(); i++) {
		ofSetColor(colors[i]);
		ofDrawCircle(x + paletteDrawSize/2, y + paletteDrawSize/2, paletteDrawSize/2);

		// draw outline if selected color
		if(colors[i] == drawColor) {
			ofPushStyle();
			ofNoFill();
			ofSetColor(255);
			ofSetLineWidth(3);
			ofDrawRectangle(x, y, paletteDrawSize, paletteDrawSize);
			ofPopStyle();
		}

		x += paletteDrawSize;

		// wrap around if doesn't fit on screen
		if(x > ofGetWidth() - paletteDrawSize) {
			x = 0;
			y += paletteDrawSize;
		}
	}

	// display drawing helpers
	ofNoFill();
	switch(drawMode) {
	case 0:
		ofSetLineWidth(3);
		ofSetColor(ofColor::black);
		ofDrawCircle(ofGetMouseX(), ofGetMouseY(), drawRadius+1);

		ofSetLineWidth(3);
		ofSetColor(drawColor);
		ofDrawCircle(ofGetMouseX(), ofGetMouseY(), drawRadius);

		break;
	case 1:
		if(ofGetMousePressed(0)) {
			ofSetLineWidth(3);
			ofSetColor(ofColor::black);
			ofDrawRectangle(mousePressPos.x-1, mousePressPos.y-1, ofGetMouseX()-mousePressPos.x+3, ofGetMouseY()-mousePressPos.y+3);

			ofSetLineWidth(3);
			ofSetColor(drawColor);
			ofDrawRectangle(mousePressPos.x, mousePressPos.y, ofGetMouseX()-mousePressPos.x, ofGetMouseY()-mousePressPos.y);
		}
	}
}


//--------------------------------------------------------------
// setup the drawing tool by folder name
void ofApp::setupDrawingTool(string model_dir) {

	// load test image
	ofLogVerbose() << "loading test image";
	ofImage img;
	img.load(ofFilePath::join(model_dir, "shoe.png"));
	if(img.isAllocated()) {
		fbo.begin();
		ofSetColor(255);
		img.draw(0, 0, fbo.getWidth(), fbo.getHeight());
		fbo.end();

	} else {
		ofLogError() << "Test image not found";
	}

	// load color palette for drawing
	ofLogVerbose() << "loading color palette";
	colors.clear();
	ofBuffer buf;
	buf = ofBufferFromFile(ofFilePath::join(model_dir, "/palette.txt"));
	if(buf.size()>0) {
		for(const auto& line : buf.getLines()) {
			ofLogVerbose() << line;
			if(line.size() == 6) // if valid hex code
				colors.push_back(ofColor::fromHex(ofHexToInt(line)));
		}
		drawColorIndex = 0;
		if(colors.size() > 0) drawColor = colors[0];
	} else {
		ofLogError() << "Palette info not found";
	}

	// load default brush info
	ofLogVerbose() << "loading default brush info";
	buf = ofBufferFromFile(ofFilePath::join(model_dir, "/default_brush.txt"));
	if(buf.size()>0) {
		auto str_info = buf.getFirstLine();
		ofLogVerbose() << str_info;
		auto str_infos = ofSplitString(str_info, " ", true, true);
		if(str_infos[0]=="draw") drawMode = 0;
		else if(str_infos[0]=="box") drawMode = 1;
		else ofLogError() << "Unknown draw mode: " << str_infos[0];

		drawRadius = ofToInt(str_infos[1]);
		ofLogError() << "draw" << drawRadius;

	} else {
		ofLogError() << "Default brush info not found";
	}

}


//--------------------------------------------------------------
// draw image or fbo etc with border and label
// typename T must have draw(x,y), isAllocated(), getWidth(), getHeight()
template <typename T>
bool ofApp::drawImage(const T& img, string label) {
	if(img.isAllocated()) {
		ofSetColor(255);
		ofFill();
		img.draw(0, 0);

		// draw border
		ofNoFill();
		ofSetColor(200);
		ofSetLineWidth(1);
		ofDrawRectangle(0, 0, img.getWidth(), img.getHeight());

		// draw label
		ofDrawBitmapString(label, 10, img.getHeight()+15);

		ofTranslate(img.getWidth(), 0);
		return true;
	}

	return false;
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch(key) {

	case 'd':
	case 'D':
		drawMode = 1 - drawMode;
		break;

	case 'c':
		if(drawRadius > 0) drawRadius--;
		break;

	case 'v':
		drawRadius++;
		break;

	case 'z':
		drawColorIndex--;
		if(drawColorIndex < 0) drawColorIndex += colors.size(); // wrap around
		drawColor = colors[drawColorIndex];
		break;

	case 'x':
		drawColorIndex++;
		if(drawColorIndex >= colors.size()) drawColorIndex -= colors.size(); // wrap around
		drawColor = colors[drawColorIndex];
		break;

	case 'i':
	case 'I':
		if(ofGetMouseX() < fbo.getWidth() && ofGetMouseY() < fbo.getHeight()) {
			drawColor = imgIn.getColor(ofGetMouseX(), ofGetMouseY());
		}
		break;

	case OF_KEY_DEL:
	case OF_KEY_BACKSPACE:
		fbo.begin();
		ofClear(255);
		fbo.end();
		break;

	case OF_KEY_RETURN:
		autoRun ^= true;
		break;
	}
}


//--------------------------------------------------------------
void ofApp::mouseDragged( int x, int y, int button) {
	switch(drawMode) {
	case 0: // draw
		fbo.begin();
		ofSetColor(drawColor);
		ofFill();
		if(drawRadius>0) {
			ofDrawCircle(x, y, drawRadius);
			ofSetLineWidth(drawRadius*2);
		} else {
			ofSetLineWidth(0.1f);
		}
		ofDrawLine(x, y, ofGetPreviousMouseX(), ofGetPreviousMouseY());
		fbo.end();
		break;
	case 1: // draw boxes
		break;
	}
}


//--------------------------------------------------------------
void ofApp::mousePressed( int x, int y, int button) {
	mousePressPos = ofVec2f(x, y);
	mouseDragged(x, y, button);
}


//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {
	switch(drawMode) {
	case 0: // draw
		break;
	case 1: // boxes
		fbo.begin();
		ofSetColor(drawColor);
		ofFill();
		ofDrawRectangle(mousePressPos.x, mousePressPos.y, x-mousePressPos.x, y-mousePressPos.y);
		fbo.end();
		break;
	}
}


//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {
	if(dragInfo.files.empty()) return;

	string file_path = dragInfo.files[0];

	// only PNGs work for some reason when Tensorflow is linked in
	ofImage img;
	img.load(file_path);
	if(img.isAllocated()) {
		fbo.begin();
		ofSetColor(255);
		img.draw(0, 0, fbo.getWidth(), fbo.getHeight());
		fbo.end();
	}
}


//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

