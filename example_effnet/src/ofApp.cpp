#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){

    // use TensorFlow ops through the cppflow wrappers
    // load a jpeg picture cast it to float and add a dimension for batches
    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("my_cat.jpg")));
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);

    // load and infer the model
    cppflow::model model(ofToDataPath("model"));
    auto output = model(input);

    // interpret the output
    auto maxLabel = cppflow::arg_max(output, 1);
    std::cout << "Maximum likelihood: " << maxLabel << std::endl;

    // access each element via the internal vector
    auto outputVector = output.get_data<float>();
    
    std::cout << "[281] tabby cat: " << outputVector[281] << std::endl;
    std::cout << "[282] tiger cat: " << outputVector[282]  << std::endl;
    std::cout << "[283] persian cat: " << outputVector[283]  << std::endl;
    std::cout << "[284] siamese cat: " << outputVector[284]  << std::endl;
    std::cout << "[285] egyptian cat: " << outputVector[285]  << std::endl;
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
