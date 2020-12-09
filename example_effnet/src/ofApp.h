#pragma once

#include <string>

#include "ofMain.h"

class ofApp : public ofBaseApp{

  std::string file;
  std::string model;

public:
    void setup();
    void update();
    void draw();

    ofApp() :
      file("/home/foo/Projects/OFx/addons/ofxTensorFlow2/example_effnet/my_cat.jpg"),
      model("/home/foo/Projects/OFx/addons/ofxTensorFlow2/example_effnet/model/")
      {
    }

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

};
