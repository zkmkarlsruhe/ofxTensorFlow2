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

      // check for frame
      if(vidGrabber.isFrameNew()){
          // get the frame
          ofPixels & pixels = vidGrabber.getPixels();
          // std::cout << "pixels: " << pixels.size() << std::endl;

          // example image
          std::vector<float> v(196608, 1);
          for(size_t i = 0; i < 196608; i++){
              v[i] = pixels[i];
          }

          // create tensor from vector
          // input = cppflow::tensor(v, {256, 256, 3});

          // create tenosr from image file
          auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("/home/foo/Projects/OFx/addons/ofxTensorFlow2/example_pix2pix/cat.jpg")));

          // cast data type and expand to batch size of 1
          input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
          input = cppflow::expand_dims(input, 0);

          // check the current tensor values
          // auto vec = input.get_data<float>();
          // for(auto it = std::begin(vec); it != std::end(vec); ++it) {
          //     std::cout << *it << std::endl;
          // }

          // start neural network and time measurement
          auto start = std::chrono::system_clock::now();
          auto output = this->model(input);
          auto output_vector = output.get_data<float>();
          auto end = std::chrono::system_clock::now();
          std::chrono::duration<double> diff = end-start;

          int i = 0;
          for(auto it = std::begin(output_vector); it != std::end(output_vector); ++it) {
              std::cout << *it  << std::endl;
              i++;
          }

        std::cout << "It's a tiger cat: " <<  cppflow::arg_max(output, 1) << std::endl;
        std::cout << "Time: " << diff.count() << std::endl;
        std::cout << "Pixels: " << i << std::endl;



          // for(size_t i = 0; i < 196608; i++){
          //     //invert the color of the pixel
          //     videoInverted[i] = output[i] * 1033055;
          // }

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
