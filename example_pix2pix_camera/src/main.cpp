#include "ofMain.h"
#include "ofApp.h"

//========================================================================

int main(int argc, char * argv[]){

    ofSetupOpenGL(1024,768,OF_WINDOW);			// <-------- setup the GL context

    ofRunApp(new ofApp("model"));

}
