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
#include "ofxTensorFlow2Utils.h"

#include "ofFileUtils.h"
#include "ofUtils.h"
#include "ofLog.h"

namespace ofxTF2 {

Model::Model(const std::string & modelPath, const cppflow::model::TYPE type) {
	this->type = type;
	Model::load(modelPath);
}

Model::~Model(){
	Model::clear();
}

void Model::setModelType(const cppflow::model::TYPE type) {
	this->type = type;
}

bool Model::load(const std::string & modelPath) {
	Model::clear();
	std::string path = ofToDataPath(modelPath);
	if (this->type == cppflow::model::SAVED_MODEL){
		if(!ofDirectory::doesDirectoryExist(path)) {
			ofLogError("ofxTensorFlow2") << "Model: model path not found: "
				<< modelPath;
			return false;
		}
	}
	else if (this->type == cppflow::model::FROZEN_GRAPH){
		if(!ofFile::doesFileExist(path)) {
			ofLogError("ofxTensorFlow2") << "Model: model path not found: "
				<< modelPath;
			return false;
		}
	}
	else {
		ofLogError("ofxTensorFlow2") << "Model: model type unknown";
			return false;
	}
	auto model = new cppflow::model(path, this->type);
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

	if (inputs.size() > inputNames_.size()) {
		ofLogError("ofxTensorFlow2") << "Model: number of inputs greater than "
			<< "number of input names";
		return std::vector<cppflow::tensor>(-1);
	}

	// add the names from settings and tensors to the vector of arguments
	for(unsigned int i = 0; i < inputs.size(); i++) {
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
	ofLogNotice("ofxTensorFlow2") << "====== Model Operations with Shapes ======";
	auto ops = model_->get_operations();
	for(const auto & el : ops) {
		if (el.compare("NoOp") != 0){
			auto s = vectorToString(model_->get_operation_shape(el));
			ofLogNotice("ofxTensorFlow2") << el << " with shape: " << s;
		}
	}
	ofLogNotice("ofxTensorFlow2") << "============ End ==============";
}

}; // end namespace ofxTF2
