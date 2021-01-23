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

#include "cppflow/cppflow.h"
#include "ofxTF2Tensor.h"
#include "ofFileUtils.h"

/// model-specific settings?
struct ofxTF2ModelSettings {
	std::vector<shape_t> inputShape;
	std::vector<shape_t> outputShape;
};

class ofxTF2Model {

public:

	/// \section Constructors

	ofxTF2Model() = default;
	ofxTF2Model(const std::string & modelPath);
	virtual ~ofxTF2Model();

	/// \section Functions

	/// load model
	/// returns true on success
	bool load(const std::string & modelPath);

	/// clear model
	void clear();

	/// set up model with settings?
	bool setup(const ofxTF2ModelSettings & settings);

	/// run model on input
	ofxTF2Tensor run(const ofxTF2Tensor & tensor) const;
    
    /// run model on input
    ofxTF2Tensor run(const cppflow::tensor & tensor) const;

protected:

	ofxTF2ModelSettings settings;
	std::string modelPath_;
	cppflow::model * model_ = nullptr;
};
