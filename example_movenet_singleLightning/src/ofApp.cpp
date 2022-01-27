/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    ofSetFrameRate(60);
    ofSetVerticalSync(true);
    //TODO
//    ofSetWindowTitle("example_movenet | singlepose lightning");
    ofSetWindowTitle("example_movenet");
    //https://tfhub.dev/google/movenet/singlepose/lightning/4
    if(!movenet.setup("model")) {
        std::exit(EXIT_FAILURE);
    }
    
#ifdef USE_LIVE_VIDEO
    // setup video grabber
    video.setDesiredFrameRate(30);
    video.setup(camWidth, camHeight);
#else
    video.load("production ID 3873059_2.mp4");
    camWidth = video.getWidth();
    camHeight = video.getHeight();
    video.play();
#endif
    
    imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::update() {
    video.update();
    if(video.isFrameNew()) {
        ofPixels pixels(video.getPixels());
        
        pixels.crop((camWidth-camHeight)/2,0,camHeight,camHeight);
        pixels.resize(nnWidth, nnHeight);
        if(mirror) {
            pixels.mirror(false, true);
        }
        imgOut.setFromPixels(pixels);
        imgOut.update();
        
        
        // feed input frame as pixels
        movenet.setInput(pixels);
    }
    
    // run model on current input frame
    movenet.update();
}

//--------------------------------------------------------------
void ofApp::draw() {

    imgOut.draw(0, 0);
    video.draw(ofGetWidth()-320,0,320,240);
 
    movenet.draw();
    ofDrawBitmapStringHighlight(ofToString((int)ofGetFrameRate()) + " fps", 4, 12);
}

//--------------------------------------------------------------
void ofApp::exit() {
    movenet.stopThread();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
    switch(key) {
        case 'm':
            // toggle camera mirroring
#ifdef USE_LIVE_VIDEO
            mirror = !mirror;
#endif
            break;
        case 'r':
            // restart video
#ifndef USE_LIVE_VIDEO
            video.stop();
            video.play();
#endif
            break;
        case 't':
            // toggle threading
            if(movenet.isThreadRunning()) {
                movenet.stopThread();
                ofLogNotice() << "stopping thread";
            }
            else {
                movenet.startThread();
                ofLogNotice() << "starting thread";
            }
            break;
    }
}
