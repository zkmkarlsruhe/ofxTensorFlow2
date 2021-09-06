#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    ofSetLineWidth(2);

    movenet.setup("model");

    #ifdef USE_LIVE_VIDEO
        // setup video grabber
        video.setDesiredFrameRate(30);
        video.setup(camWidth, camHeight);
	    imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
    #else
        video.load("production ID 3873059_2.mp4");
        video.play();
    #endif
}

//--------------------------------------------------------------
void ofApp::update(){


    video.update();
    if(video.isFrameNew()) {
        ofPixels pixels(video.getPixels());
        #ifdef USE_LIVE_VIDEO
            pixels.resize(nnWidth, nnHeight);
            imgOut.setFromPixels(pixels);
            imgOut.update();
        #endif
        movenet.update(pixels);
    }


}

//--------------------------------------------------------------
void ofApp::draw(){

    #ifdef USE_LIVE_VIDEO
        imgOut.draw(0,0);
    #else
        video.draw(0,0);
    #endif

    movenet.draw();

    ofDrawBitmapStringHighlight("fps:: " + ofToString((int)ofGetFrameRate()), glm::vec2(20,20));

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if(key = 'r') {
        #ifndef USE_LIVE_VIDEO
            video.stop();
            video.play();
        #endif
    }
}
