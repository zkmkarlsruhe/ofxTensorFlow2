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

ofxTF2ThreadedModel::~ofxTF2ThreadedModel() {
	waitForThread();
}

bool ofxTF2ThreadedModel::readyForInput() {
	bool ret = false;
	if(tryLock()) {
		ret = !newInput_;
		unlock();
	}
	return ret;
}

bool ofxTF2ThreadedModel::update(cppflow::tensor input) {
	bool ret = false;
	if(tryLock()) {
		input_ = input;
		newInput_ = true;
		unlock();
		ret = true;
	}
	return ret;
}

bool ofxTF2ThreadedModel::isOutputNew() {
	bool ret = false;
	if(tryLock()) {
		ret = newOutput_;
		unlock();
	}
	return ret;
}

cppflow::tensor ofxTF2ThreadedModel::getOutput() {
	cppflow::tensor ret;
	lock();
	ret = output_;
	newOutput_ = false;
	unlock();
	return ret;
}

void ofxTF2ThreadedModel::setIdleTime(unsigned int ms) {
	idleMS_ = ms;
}


void ofxTF2ThreadedModel::loadSafely(const std::string & modelPath) {
	lock();
	load(modelPath);
	unlock();
}

void ofxTF2ThreadedModel::clearSafely() {
	lock();
	clear();
	unlock();
}


// ==== protected ====

void ofxTF2ThreadedModel::threadedFunction() {
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
