#include "ofMain.h"
#include "ofApp.h"

//========================================================================

int main(int argc, char * argv[]){

    ofSetupOpenGL(1024,768,OF_WINDOW);			// <-------- setup the GL context

    ofRunApp(new ofApp("python/model1"));
    // ofRunApp(new ofApp("/home/foo/Projects/OFx/addons/ofxTensorFlow2/example_effnet/model"));

}
