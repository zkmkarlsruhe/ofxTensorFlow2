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

namespace ofxTF2 {

Model::Model(const std::string & modelPath) {
	Model::load(modelPath);
}

Model::~Model(){
	Model::clear();
}

bool Model::load(const std::string & modelPath) {
	Model::clear();
	std::string path = ofToDataPath(modelPath);
	if(!ofDirectory::doesDirectoryExist(path)) {
		ofLogError("ofxTensorFlow2") << "Model: model path not found: "
			<< modelPath;
		return false;
	}
	auto model = new cppflow::model(path);
	if(!model) {
		modelPath_ = "";
		ofLogError("ofxTensorFlow2") << "Model: model could not be initialized!";
		return false;
	}	
	model_ = model;
	modelPath_ = modelPath;
	ofLogVerbose("ofxTensorFlow2") << "Model: loaded model: " << modelPath_;
	return true;
}

void Model::setup(const std::vector<std::string> & inputNames,
                  const std::vector<std::string> & outputNames) {
	inputNames_ = inputNames;
	outputNames_ = outputNames;
}

void Model::clear() {
	if(model_) {
		ofLogVerbose("ofxTensorFlow2") << "Model: clear model: " << modelPath_;
		delete model_;
		model_ = nullptr;
		modelPath_ = "";
	}
}

cppflow::tensor Model::runModel(const cppflow::tensor & input) const {
	return Model::runMultiModel(std::vector<cppflow::tensor>{input})[0];
}

std::vector<cppflow::tensor>
Model::runMultiModel(const std::vector<cppflow::tensor> & inputs) const {
	
	// define the type of an input argument and a vector of it
	using inputTuple_t = std::tuple<std::string, cppflow::tensor>;
	std::vector<inputTuple_t> inputArguments;

	// add the names from settings and tensors to the vector of arguments
	for(unsigned int i = 0; i < inputNames_.size(); i++) {
		inputArguments.push_back(inputTuple_t(inputNames_[i], inputs[i]));
	}

	// if model exists run it with multiple inputs and outputs
	if(model_) {
		return (*model_)(inputArguments, outputNames_);
	}
	else {
        ofLogError("ofxTensorFlow2") << "Model: no model loaded! "
			<< "Returning tensor containing -1.";
		return std::vector<cppflow::tensor>(-1);
	}
}

bool Model::isLoaded() {
	return model_ != nullptr;
}

void Model::printOperations() {
	ofLogNotice("ofxTensorFlow2") << "====== Model Operations ======";
	auto ops = model_->get_operations();
	for(auto & el : ops) {
		ofLogNotice("ofxTensorFlow2") << el;
	}
	ofLogNotice("ofxTensorFlow2") << "============ End ==============";
}

}; // end namespace ofxTF2
