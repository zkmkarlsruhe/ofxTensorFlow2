#include "ofApp.h"

//--------------------------------------------------------------
ofApp::ofApp(std::string modelPath)
: model(ofToDataPath(modelPath))
    {}


//--------------------------------------------------------------
void ofApp::setup(){

    nnWidth = 512;
    nnHeight = 512;
    
    imgIn.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
    imgOut.allocate(nnWidth, nnHeight, OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::update(){

    // create tensor from image file
    input = cppflow::decode_jpeg(cppflow::read_file(std::string("cat3.jpg")));
    
    // cast data type and expand to batch size of 1
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);

    // start neural network and time measurement
    auto start = std::chrono::system_clock::now();
    output = this->model(input);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count() << std::endl;

    auto outputVector = output.get_data<float>();
    auto inputVector = input.get_data<float>();

    auto & pixels = imgOut.getPixels();
    for(int i=0; i<pixels.size(); i++) pixels[i] = outputVector[i];

    auto & pixels_in = imgIn.getPixels();
    for(int i=0; i<pixels_in.size(); i++) pixels_in[i] = inputVector[i];

    imgOut.update();
    imgIn.update();
}

//--------------------------------------------------------------
void ofApp::draw() {
    imgIn.draw(0, 0);
    imgOut.draw(nnWidth, nnHeight, nnWidth * 2, nnHeight * 2);
}
