/*
 * ofxTensorFlow2
 *
 * Copyright (c) 2021 ZKM | Hertz-Lab
 * Paul Bethge <bethge@zkm.de>
 * Dan Wilcox <dan.wilcox@zkm.de>
 *
 * BSD Simplified License.
 * For information on usage and redistribution, and for a DISCLAIMER OF ALL
 * WARRANTIES, see the file, "LICENSE.txt," in this distribution.
 *
 * This code has been developed at ZKM | Hertz-Lab as part of „The Intelligent 
 * Museum“ generously funded by the German Federal Cultural Foundation.
 */


#include "ofxTF2Model.h"

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

ofxTF2Tensor ofxTF2Model::run(const cppflow::tensor & tensor) const {
	if (model_){
		return (*model_)(tensor);
	}
	else{
		ofLog() << "ofxTF2Model: no model loaded! Returning tensor containing -1.";
		return ofxTF2Tensor(-1);
	}
}

ofxTF2Tensor ofxTF2Model::run(const ofxTF2Tensor & tensor) const {
	if (model_){
		return (*model_)(tensor.getTensor());
	}
	else{
		ofLog() << "ofxTF2Model: no model loaded! Returning tensor containing -1.";
		return ofxTF2Tensor(-1);
	}
}

bool ofxTF2Model::setup(ofxTF2ModelSettings & settings){
	return true;
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