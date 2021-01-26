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

/// \class ofxTF2Model
/// \brief a wrapper for cppflow::model which processes input to output
///
/// basic usage example:
///
///     class ofApp : public ofBaseApp {
///     public:
///     ...
///         ofxTF2Model model;
///     };
///
///     void ofApp::setup() {
///         ...
///         model.load("path/to/modeldir");
///     }
///
///     void ofApp.cpp::update() {
///         cppflow::tensor input = ...
///         ... prepare input
///         cppflow::tensor output = model.runModel(input);
///         ... process output
///     }
///
class ofxTF2Model {

public:

	ofxTF2Model() = default;
	ofxTF2Model(const std::string & modelPath);
	virtual ~ofxTF2Model();

	/// load model
	/// TODO: describe expected model folder layout?
	/// returns true on success
	bool load(const std::string & modelPath);

	/// clear model
	void clear();
    
    /// run model on input
    cppflow::tensor runModel(const cppflow::tensor & tensor) const;

    /// returns true if model is loaded
    bool isLoaded();

protected:

	/// preprocess input tensor, called before running model
	/// implement in a subclass, default implementation simply returns input
	virtual cppflow::tensor preprocess(const cppflow::tensor & input) const;

	/// postprocess output tensor, called after running model
	/// implement in a subclass, default implementation simply returns output
	virtual cppflow::tensor postprocess(const cppflow::tensor & output) const;

	std::string modelPath_ = "";
	cppflow::model * model_ = nullptr;
};
