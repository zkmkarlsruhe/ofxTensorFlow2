#pragma once

#include "cppflow/cppflow.h"
#include "ofxTF2Tensor.h"

class ofxTF2ModelSettings {
	// left for future implementation
};

class ofxTF2Model {

	public: 

	ofxTF2Model() = default;
	ofxTF2Model(const std::string & modelPath);
	~ofxTF2Model();

	ofxTF2Tensor run(const ofxTF2Tensor & tensor);
	bool load(const std::string & modelPath);
	bool setup(const ofxTF2ModelSettings & settings);

	private:

	ofxTF2ModelSettings settings;
	std::string modelPath_;
	cppflow::model * model_;
};

ofxTF2Model::ofxTF2Model(const std::string & modelPath) {

	// todo check if is_directory()
	if (false){
		ofLog() << "ofxTF2Model: path not a folder!";
	}
	else {
		model_ = new cppflow::model(modelPath);
		if (!model_){
			ofLog() << "ofxTF2Model: model not initialized!";
		}
		else {
			ofLog() << "ofxTF2Model: loaded model: " << modelPath;
			modelPath_ = modelPath;
		}
	}
}

ofxTF2Model::~ofxTF2Model(){
	delete model_;
	ofLog() << "ofxTF2Model: destruct model: " << modelPath_;
}

ofxTF2Tensor ofxTF2Model::run(const ofxTF2Tensor & tensor){
	if (model_){
		return (*model_)(tensor);
	}
	else{
		ofLog() << "ofxTF2Model: no model loaded! Returning tensor containing -1.";
		return ofxTF2Tensor(-1);
	}
}

bool ofxTF2Model::setup(const ofxTF2ModelSettings & settings){

}

bool ofxTF2Model::load(const std::string & modelPath) {
	if (model_){
		ofLog() << "ofxTF2Model: delete current model: " << modelPath_;
		delete model_;
	}
	model_ = new cppflow::model(modelPath);

	if (!model_){
		ofLog() << "ofxTF2Model: model not initialized!";
		return false;
	}
	else {
		ofLog() << "ofxTF2Model: loaded model: " << modelPath;
		modelPath_ = modelPath;
		return true;
	}
}