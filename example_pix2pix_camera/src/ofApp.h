#pragma once

#include <string>
#include <vector>
#include "ofMain.h"
#include <cstdlib>

#include "cppflow/cppflow.h"

class ofApp : public ofBaseApp{

public:

    cppflow::model model;
    cppflow::tensor input;
    cppflow::tensor output;
    int nnWidth;
    int nnHeight;



    ofVideoGrabber vidGrabber;
    ofPixels videoInverted;
    ofTexture videoTexture;
    int camWidth;
    int camHeight;

    ofApp(std::string model_path)
    : model(model_path)
      {

      }

    void setup(){

      // ===== Camera ===== //

      camWidth = 640;  // try to grab at this size.
      camHeight = 480;

      nnWidth = 256;
      nnHeight = 256;

      vidGrabber.setDeviceID(0);
      vidGrabber.setDesiredFrameRate(25);
      vidGrabber.initGrabber(camWidth, camHeight);

      videoInverted.allocate(nnWidth, nnHeight, OF_PIXELS_RGB);
      videoTexture.allocate(videoInverted);
      ofSetVerticalSync(true);
    }

    void update(){
      ofBackground(100, 100, 100);
      vidGrabber.update();

      // check for new frame
      if(vidGrabber.isFrameNew()){
          // get the frame
          ofPixels & pixels = vidGrabber.getPixels();
          // std::cout << "pixels: " << pixels.size() << std::endl;

          // example image
          std::vector<float> v(196608, 1);
          for(size_t i = 0; i < 196608; i++){
              v[i] = pixels[i];
          }


          //load the inverted pixels
          videoTexture.loadData(videoInverted);
      }


    }

    void draw() {

          ofSetHexColor(0xffffff);
          vidGrabber.draw(20, 20);
          videoTexture.draw(20 + camWidth, 20, nnHeight, nnHeight);


    }

    void keyPressed(int key){
      if(key == 's' || key == 'S'){
          vidGrabber.videoSettings();
      }
    }

    void keyReleased(int key){}
    void mouseMoved(int x, int y ){}
    void mouseDragged(int x, int y, int button){}
    void mousePressed(int x, int y, int button){}
    void mouseReleased(int x, int y, int button){}
    void mouseEntered(int x, int y){}
    void mouseExited(int x, int y){}
    void windowResized(int w, int h){}
    void dragEvent(ofDragInfo dragInfo){}
    void gotMessage(ofMessage msg){}

};
