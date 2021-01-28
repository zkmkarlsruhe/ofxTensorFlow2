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
/// to add custom built-in pre or post processing, subclass and override
/// the runModel() virtual function:
///
///     class Model : public ofxTF2Model {
///     public:
///         cppflow::tensor runModel(const cppflow::tensor & input) const;
///     };
///
///     cppflow::tensor Model::runModel(const cppflow::tensor & input) const {
///         input = input * cppflow::tensor({-1}); // invert, etc
///         ... preprocess input
///         cppflow::tensor output = ofxTF2Model::runModel(input); // call super
///         ... postprocess output
///         output = output * cppflow::tensor({-1}); // invert back, etc
///         return output; // done
///     }
///
class ofxTF2Model {

public:

	ofxTF2Model() = default;
	ofxTF2Model(const std::string & modelPath);
	virtual ~ofxTF2Model();

	/// load a SavedModel directory relative to bin/data
	/// directories for SavedModels includes assets, variables and a .pb file
	/// returns true on success
	bool load(const std::string & modelPath);

	/// clear model
	void clear();
    
    /// run model on input, blocks until returning output
    /// implement in a subclass to add custom pre / post processing
    virtual cppflow::tensor runModel(const cppflow::tensor & input) const;

    /// returns true if model is loaded
    bool isLoaded();

protected:

	std::string modelPath_ = "";
	cppflow::model * model_ = nullptr;
};
