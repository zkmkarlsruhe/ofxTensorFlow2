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

/// \class Model
/// \brief a wrapper for cppflow::model which processes input to output
///
/// basic usage example:
///
///     class ofApp : public ofBaseApp {
///     public:
///     ...
///         MyModel model;
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
///     class MyModel : public ofxTF2::Model {
///     public:
///         cppflow::tensor runModel(const cppflow::tensor & input) const;
///     };
///
///     cppflow::tensor MyModel::runModel(const cppflow::tensor & input) const {
///         input = input * cppflow::tensor({-1}); // invert, etc
///         ... preprocess input
///         cppflow::tensor output = Model::runModel(input); // call super
///         ... postprocess output
///         output = output * cppflow::tensor({-1}); // invert back, etc
///         return output; // done
///     }
///

namespace ofxTF2 {


struct ModelSettings {
	std::string inputName_;
	std::string outputName_;
	// initialize members with TF default values
	ModelSettings() : inputName_ ({"serving_default_input_1"}), 
					outputName_ ({"StatefulPartitionedCall"}) { }
};


class Model {

public:	

	Model() = default;
	Model(const std::string & modelPath);
	virtual ~Model();

	/// load a SavedModel directory relative to bin/data
	/// directories for SavedModels include assets, variables, and a .pb file
	/// returns true on success
	virtual bool load(const std::string & modelPath);

	// set parameters like in and ouput names
	void setup(const ModelSettings & settings);

	/// clear model
	virtual void clear();

	/// run model on input, blocks until returning output
	/// implement in a subclass to add custom pre / post processing
	virtual cppflow::tensor runModel(const cppflow::tensor & input) const;

	// print the signature
	void printOps();

	/// returns true if model is loaded
	bool isLoaded();

protected:

	ModelSettings settings_;
	std::string modelPath_ = "";
	cppflow::model * model_ = nullptr;
};

}; // end namespace ofxTF2
