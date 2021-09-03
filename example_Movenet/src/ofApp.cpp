#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    ofSetLineWidth(2);

    movenet.setup("model");

    video.load("production ID 3873059_2.mp4");
    video.play();
}

//--------------------------------------------------------------
void ofApp::update(){

    video.update();

    if(video.isFrameNew()){

        movenet.update(video.getPixels());
    }

}

//--------------------------------------------------------------
void ofApp::draw(){

    video.draw(0,0);
    movenet.draw();

    ofDrawBitmapStringHighlight("fps:: " + ofToString((int)ofGetFrameRate()), glm::vec2(20,20));

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if(key = 'r') {
        video.stop();
        video.play();
    }
}
