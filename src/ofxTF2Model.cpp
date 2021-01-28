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

#include "ofFileUtils.h"
#include "ofUtils.h"
#include "ofLog.h"

ofxTF2Model::ofxTF2Model(const std::string & modelPath) {
	load(modelPath);
}

ofxTF2Model::~ofxTF2Model(){
	clear();
}

bool ofxTF2Model::load(const std::string & modelPath) {
	clear();
	std::string path = ofToDataPath(modelPath);
	if (!ofDirectory::doesDirectoryExist(path)){
		ofLogError() << "ofxTF2Model: model path not found: " << modelPath;
		return false;
	}
	auto model = new cppflow::model(path);
	if (!model){
		modelPath_ = "";
		ofLogError() << "ofxTF2Model: model could not be initialized!";
		return false;
	}	
	model_ = model;
	modelPath_ = modelPath;
	ofLogVerbose() << "ofxTF2Model: loaded model: " << modelPath_;
	return true;
}


void ofxTF2Model::clear() {
	if (model_){
		ofLogVerbose() << "ofxTF2Model: clear model" << modelPath_;
		delete model_;
		model_ = nullptr;
		modelPath_ = "";
	}
}

cppflow::tensor ofxTF2Model::runModel(const cppflow::tensor & input) const {
	if (model_){
		return (*model_)(input);
	}
	else{
		ofLog() << "ofxTF2Model: no model loaded! Returning tensor containing -1.";
		return cppflow::tensor(-1);
	}
}

bool ofxTF2Model::isLoaded() {
	return model_ != nullptr;
}
