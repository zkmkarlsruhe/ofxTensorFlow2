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

void ThreadedModel::clear() {
	lock();
	Model::clear();
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

bool ThreadedModel::update(cppflow::tensor input) {
	bool ret = false;
	if(tryLock()) {
		input_ = input;
		newInput_ = true;
		unlock();
		ret = true;
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

cppflow::tensor ThreadedModel::getOutput() {
	cppflow::tensor ret;
	lock();
	ret = output_;
	newOutput_ = false;
	unlock();
	return ret;
}

void ThreadedModel::setIdleTime(unsigned int ms) {
	idleMS_ = ms;
}

cppflow::tensor ThreadedModel::runModel(const cppflow::tensor & input) const{
	return Model::runModel(input);
}

// ==== protected ====

void ThreadedModel::threadedFunction() {
	while(isThreadRunning()) {
		lock();
		if(newInput_) {
			output_ = runModel(input_);
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
