#pragma once

#include <string>
#include <vector>
#include "ofMain.h"
#include <cstdlib>

#include "cppflow/cppflow.h"

class ofApp : public ofBaseApp{

public:

    cppflow::model model;
    int nnWidth;
    int nnHeight;

    ofPixels frameProcessed;
    ofVideoGrabber videoGrabber;
    ofTexture inputTexture;
    ofTexture outputTexture;
    int camWidth;
    int camHeight;

    //--------------------------------------------------------------
    ofApp(std::string modelPath)
    : model(modelPath)
      {}

    void setup(){
    
      camWidth = 640;  // try to grab at this size.
      camHeight = 480;

      nnWidth = 512;
      nnHeight = 512;

      videoGrabber.setDeviceID(0);
      videoGrabber.setDesiredFrameRate(30);
      videoGrabber.initGrabber(camWidth, camHeight);

      frameProcessed.allocate(nnWidth, nnHeight, OF_PIXELS_RGB);
      inputTexture.allocate(frameProcessed);
      outputTexture.allocate(frameProcessed);
      ofSetVerticalSync(true);
    }

    void update(){
      ofBackground(100, 100, 100);
      videoGrabber.update();

      // check for new frame
      if(videoGrabber.isFrameNew()){

        // get the frame
        ofPixels & pixels = videoGrabber.getPixels();

        // resize pixels
        ofPixels resizedPixels(pixels);
        resizedPixels.resize(nnWidth, nnHeight);

        // copy to tensor
        cppflow::tensor input(
              std::vector<float>(resizedPixels.begin(),
                                  resizedPixels.end()),
                                  {nnWidth, nnHeight, 3});

        // cast data type and expand to batch size of 1
        input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        input = cppflow::expand_dims(input, 0);
        input = cppflow::div(input, cppflow::tensor({127.5f}));
        input = cppflow::add(input, cppflow::tensor({-1.0f}));

        // auto & input_vector = input.get_data<float>();
        // for(int i=0; i<input_vector.size(); i++) input_vector[i] = input_vector[i] / 255;

        // start neural network and time measurement
        auto start = std::chrono::system_clock::now();
        auto output = this->model(input);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end-start;

        // std::cout << output << std::endl;
        std::cout << "Time: " << diff.count() << std::endl;

        // copy to output frame and postprocessing
        auto outputVector = output.get_data<float>();
        for(int i=0; i<outputVector.size(); i++) frameProcessed[i] = (outputVector[i] + 1) * 127.5;

        outputTexture.loadData(frameProcessed);
        inputTexture.loadData(resizedPixels);
        }
    }

    void draw() {
        ofSetHexColor(0xffffff);
        videoGrabber.draw(20, 20);
        outputTexture.draw(20 + camWidth, 20, nnWidth, nnHeight);
        inputTexture.draw(20, 20 + camHeight, nnWidth, nnHeight);
    }

    void keyPressed(int key){
      if(key == 's' || key == 'S'){
          videoGrabber.videoSettings();
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
