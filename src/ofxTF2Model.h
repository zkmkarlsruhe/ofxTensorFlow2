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


#pragma once

#include "ofMain.h"
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

	ofxTF2Tensor run(const cppflow::tensor & tensor) const;
	ofxTF2Tensor run(const ofxTF2Tensor & tensor) const;
	bool load(const std::string & modelPath);
	bool setup(ofxTF2ModelSettings & settings);

	private:

	ofxTF2ModelSettings settings;
	std::string modelPath_;
	cppflow::model * model_;
};

