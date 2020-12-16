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
    int nn_input_size;



    ofPixels frame_processed;
    ofVideoGrabber video_grabber;
    ofTexture video_nn_input;
    ofTexture video_texture;
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

      nnWidth = 512;
      nnHeight = 512;

      nn_input_size = nnWidth * nnHeight * 3;

      video_grabber.setDeviceID(0);
      video_grabber.setDesiredFrameRate(30);
      video_grabber.initGrabber(camWidth, camHeight);

      frame_processed.allocate(nnWidth, nnHeight, OF_PIXELS_RGB);
      video_nn_input.allocate(frame_processed);
      video_texture.allocate(frame_processed);
      ofSetVerticalSync(true);
    }

    void update(){
      ofBackground(100, 100, 100);
      video_grabber.update();

      // check for new frame
      if(video_grabber.isFrameNew()){

        // get the frame
        ofPixels & pixels = video_grabber.getPixels();
        // std::cout << "pixels: " << pixels.size() << std::endl;

        // resize pixels
        ofPixels pixels_resized(pixels);
        pixels_resized.resize(nnWidth, nnHeight);

        // copy to std vector
        // std::vector<float> v(nn_input_size, 1);
        // for(size_t i = 0; i < nn_input_size; i++){
        //     pixels_resized[i] = (pixels_resized[i] / 127.5) - 1;
        // }

        // copy to tensor
        cppflow::tensor input(
              std::vector<float>(pixels_resized.begin(),
                                  pixels_resized.end()),
                                  {256, 256, 3});

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

        // copy to output frame
        auto output_vector = output.get_data<float>();
        for(int i=0; i<output_vector.size(); i++) frame_processed[i] = (output_vector[i] + 1) * 127.5;

        //load the inverted pixels
        video_texture.loadData(frame_processed);
        video_nn_input.loadData(pixels_resized);
        }
    }

    void draw() {
        ofSetHexColor(0xffffff);
        video_grabber.draw(20, 20);
        video_texture.draw(20 + camWidth, 20, nnHeight * 2, nnHeight * 2);
        video_nn_input.draw(20, 20 + camHeight, nnHeight, nnHeight);
    }

    void keyPressed(int key){
      if(key == 's' || key == 'S'){
          video_grabber.videoSettings();
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
