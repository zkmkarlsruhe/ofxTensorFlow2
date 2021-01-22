#include "ofxTF2Model.h"

// ==== constructors ====

ofxTF2Model::ofxTF2Model(const std::string & modelPath) {
	load(modelPath);
}

ofxTF2Model::~ofxTF2Model(){
	clear();
}

// ==== functions ====

bool ofxTF2Model::load(const std::string & modelPath) {
	clear();
	std::string path = ofToDataPath(modelPath);
	if (!ofDirectory::doesDirectoryExist(path)){
		ofLogError() << "ofxTF2Model: model path not found: " << modelPath;
		return false;
	}
	model_ = new cppflow::model(path);
	if (!model_){
		ofLogError() << "ofxTF2Model: model could not be initialized!";
		return false;
	}
	ofLogVerbose() << "ofxTF2Model: loaded model: " << modelPath;
	modelPath_ = path;
	return true;
}

void ofxTF2Model::clear() {
	if (model_){
		delete model_;
		model_ = nullptr;
		ofLogVerbose() << "ofxTF2Model: cleared model: " << modelPath_;
	}
	modelPath_ = "";
}

bool ofxTF2Model::setup(const ofxTF2ModelSettings & settings) {}

ofxTF2Tensor ofxTF2Model::run(const ofxTF2Tensor & tensor) {
	if (model_){
		return ofxTF2Tensor((*model_)(tensor));
	}
	else{
		ofLogWarning() << "ofxTF2Model: no model loaded! Returning tensor containing -1.";
		return ofxTF2Tensor(-1);
	}
}
