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

    ofFbo fbo;


    ofImage img_out;
    ofImage img_in;


    ofApp(std::string model_path)
    : model(model_path)
      {}

    void setup(){

        fbo.allocate(ofGetWidth(), ofGetHeight());
        fbo.begin();
        ofClear(255);
        fbo.end();


		nnWidth = 512;
		nnHeight = 512;
        
		img_in.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
        img_out.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
    }

    void update(){

        // create tensor from image file
        input = cppflow::decode_jpeg(cppflow::read_file(std::string("cat3.jpg")));
        
		// cast data type and expand to batch size of 1
        input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        input = cppflow::expand_dims(input, 0);


        // start neural network and time measurement
        auto start = std::chrono::system_clock::now();
        auto output = this->model(input);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end-start;
        std::cout << "Time: " << diff.count() << std::endl;

        auto output_vector = output.get_data<float>();
        auto input_vector = input.get_data<float>();

        auto & pixels = img_out.getPixels();
        for(int i=0; i<pixels.size(); i++) pixels[i] = output_vector[i];

        auto & pixels_in = img_in.getPixels();
        for(int i=0; i<pixels_in.size(); i++) pixels_in[i] = input_vector[i];

        img_out.update();
        img_in.update();

//        int radius = i % 10000;
//        fbo.begin();
//        ofDrawCircle(0, 0, radius);
//        img_out.draw(0, 0);
//        fbo.end();
//        i++;
    }

    void draw() {
//        fbo.draw(0, 0);
        img_in.draw(0, 0);
        img_out.draw(nnWidth, nnHeight, nnWidth * 2, nnHeight * 2);
    }

    void keyPressed(int key){
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
