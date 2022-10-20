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

#include "ofxTF2ThreadedModel.h"

namespace ofxTF2 {

ThreadedModel::~ThreadedModel() {
	waitForThread();
}

bool ThreadedModel::load(const std::string & modelPath) {
	bool ret = false;
	lock();
	ret = Model::load(modelPath);
	unlock();
	return ret;
}

void ThreadedModel::loadAsync(const std::string & modelPath) {
	lock();
	newModelPath_ = modelPath;
	unlock();
}

void ThreadedModel::setup(const std::vector<std::string> & inputNames,
                          const std::vector<std::string> & outputNames) {
	lock();
	Model::setup(inputNames, outputNames);
	unlock();
}

void ThreadedModel::clear() {
	lock();
	Model::clear();
	unlock();
}

cppflow::tensor ThreadedModel::runModel(const cppflow::tensor & input) const{
	return Model::runModel(input);
}

std::vector<cppflow::tensor>
ThreadedModel::runMultiModel(const std::vector<cppflow::tensor> & inputs) const {
	return Model::runMultiModel(inputs);
}

bool ThreadedModel::isLoaded() {
	bool ret = false;
	lock();
	ret = Model::isLoaded();
	unlock();
	return ret;
}

void ThreadedModel::printOperations() {
	lock();
	Model::printOperations();
	unlock();
}

bool ThreadedModel::readyForInput() {
	bool ret = false;
	if(tryLock()) {
		ret = !newInput_;
		unlock();
	}
	return ret;
}

bool ThreadedModel::isOutputNew() {
	bool ret = false;
	if(tryLock()) {
		ret = newOutput_;
		unlock();
	}
	return ret;
}

bool ThreadedModel::update(const cppflow::tensor & input) {
	return ThreadedModel::update(std::vector<cppflow::tensor> {input});
}

bool ThreadedModel::update(const std::vector<cppflow::tensor> & inputs) {
	bool ret = false;
	if(tryLock()) {
		inputs_ = inputs;
		newInput_ = true;
		unlock();
		ret = true;
	}
	return ret;
}

cppflow::tensor ThreadedModel::getOutput() {
	return getOutputs()[0];
}

std::vector<cppflow::tensor> ThreadedModel::getOutputs() {
	std::vector<cppflow::tensor> ret;
	lock();
	ret = outputs_;
	newOutput_ = false;
	unlock();
	return ret;
}

void ThreadedModel::setIdleTime(unsigned int ms) {
	idleMS_ = ms;
}

// ==== protected ====

void ThreadedModel::threadedFunction() {
	while(isThreadRunning()) {
		lock();
		if(newModelPath_ != "") {
			// async model loading
			Model::load(newModelPath_);
			newModelPath_ = ""; // done
		}
		if(newInput_) {
			// if we have only one in and output use the simpler runModel function
			// we dont call runMultiModel as we want users to augment runModel
			if(inputNames_.size() <= 1 && outputNames_.size() <= 1) {
				outputs_ = {runModel(inputs_[0])};
			}
			else {
				// otherwise use runMultiModel function
				outputs_ = runMultiModel(inputs_);
			}
			newInput_ = false;
			newOutput_ = true;
			unlock();
		}
		else { // idle
			unlock();
			sleep(idleMS_);
		}
	}
}

}; // end namespace ofxTF2
