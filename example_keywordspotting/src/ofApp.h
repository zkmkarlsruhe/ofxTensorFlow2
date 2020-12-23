#pragma once

#include <string>
#include <vector>

#include <cstdlib>

#include "ofMain.h"
#include "cppflow/cppflow.h"

#include "labels.h"


class ofApp : public ofBaseApp{

public:

    // neural network
    cppflow::model model;
    cppflow::tensor input;
    cppflow::tensor output;
    
    // sound
    int bufferSize;
		ofSoundStream soundStream;
 
    //--------------------------------------------------------------
    ofApp(std::string model_path)
    : model(model_path)
      {}

    //--------------------------------------------------------------
    void setup(){

      ofSetVerticalSync(true);
      soundStream.printDeviceList();
      ofSoundStreamSettings settings;	
      // if you want to set the device id to be different than the default
      // auto devices = soundStream.getDeviceList();
      // settings.device = devices[4];

      // you can also get devices for an specific api
      // auto devices = soundStream.getDevicesByApi(ofSoundDevice::Api::PULSE);
      // settings.device = devices[0];

      // or get the default device for an specific api:
      // settings.api = ofSoundDevice::Api::PULSE;

      // or by name
      auto devices = soundStream.getMatchingDevices("default");
      if(!devices.empty()){
        settings.setInDevice(devices[0]);
      }

    	bufferSize = 16000;

      settings.setInListener(this);
      settings.sampleRate = 16000;
      settings.numOutputChannels = 0;
      settings.numInputChannels = 1;
      settings.bufferSize = bufferSize;
      soundStream.setup(settings);

      // warm up 
      input = cppflow::fill({1, 16000}, 1.0f);
      output = model(input);
      auto output_vector = output.get_data<float>();

      int argMax = std::distance(output_vector.begin(), std::max_element(output_vector.begin(), output_vector.end()));
      std::cout << "Label: " << labels_map[argMax] << std::endl;
    }

    void update(){

    }

    void draw() {

    }

    void keyPressed(int key){
      if( key == 's' ){
        soundStream.start();
      }
      
      if( key == 'e' ){
        soundStream.stop();
      }
    }

    //--------------------------------------------------------------
    void audioIn(ofSoundBuffer & buffer){

      // std::cout << "ms: " << buffer.getDurationMS() << std::endl;
      // std::cout << "frame: " << buffer.getNumFrames() << std::endl;
      auto frames = buffer.getBuffer();
      // for (size_t i = 0; i < input.size(); i++){
      //   input[i] = frames[i];
      // }

      cppflow::tensor audio(frames, {1, 16384});
      auto output_vector = model(audio).get_data<float>();
      int argMax = std::distance(output_vector.begin(), std::max_element(output_vector.begin(), output_vector.end()));
      std::cout << "Label: " << labels_map[argMax] << std::endl;
      std::cout.flush();

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
