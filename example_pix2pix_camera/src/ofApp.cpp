#include "ofApp.h"


ofApp::ofApp(std::string modelPath)
    : model(ofToDataPath(modelPath)){}


void ofApp::setup(){

    camWidth = 640;  // try to grab at this size.
    camHeight = 480;

    nnWidth = 256;
    nnHeight = 256;

    videoGrabber.setDeviceID(0);
    videoGrabber.setDesiredFrameRate(30);
    videoGrabber.initGrabber(camWidth, camHeight);

    frameProcessed.allocate(nnWidth, nnHeight, OF_PIXELS_RGB);
    inputTexture.allocate(frameProcessed);
    outputTexture.allocate(frameProcessed);
    ofSetVerticalSync(true);
}

void ofApp::update(){
    ofBackground(100, 100, 100);
    videoGrabber.update();

    // check for new frame
    if(videoGrabber.isFrameNew()){

    // get the frame
    ofPixels & pixels = videoGrabber.getPixels();
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

    // start neural network and time measurement
    auto start = std::chrono::system_clock::now();
    auto output = this->model(input);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;

    // std::cout << output << std::endl;
    std::cout << "Time: " << diff.count() << std::endl;

    // copy to output frame and postprocessing
    auto outputVector = output.get_data<float>();
    // for(int i=0; i<outputVector.size(); i++) frameProcessed[i] = (outputVector[i] + 1) * 127.5;
    for(int i=0; i<outputVector.size(); i++) frameProcessed[i] = ((outputVector[i] * 0.5 )+ 0.5) * 255;

    outputTexture.loadData(frameProcessed);
    inputTexture.loadData(resizedPixels);
    }
}

void ofApp::draw() {
    ofSetHexColor(0xffffff);
    videoGrabber.draw(20, 20);
    outputTexture.draw(20 + camWidth, 20, nnWidth, nnHeight);
    inputTexture.draw(20, 20 + camHeight, nnWidth, nnHeight);
}
