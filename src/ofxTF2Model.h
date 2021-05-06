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

namespace ofxTF2 {

/// \class Model
/// \brief a wrapper for cppflow::model which processes inputs to outputs
///
/// basic usage example:
///
/// class ofApp : public ofBaseApp {
/// public:
/// ...
///     ofxTF2::Model model;
/// };
///
/// void ofApp::setup() {
///     ...
///     model.load("path/to/modeldir");
/// }
///
/// void ofApp.cpp::update() {
///     cppflow::tensor input = ...
///     ... prepare input
///     cppflow::tensor output = model.runModel(input);
///     ... process output
/// }
///
/// to add custom built-in pre- and/or postprocessing, subclass and override
/// the runModel() virtual function:
///
/// class MyModel : public ofxTF2::Model {
/// public:
///     cppflow::tensor runModel(const cppflow::tensor & input) const;
/// };
///
/// cppflow::tensor MyModel::runModel(const cppflow::tensor & input) const {
///     input = input * cppflow::tensor({-1}); // invert, etc
///     ... preprocess input
///     cppflow::tensor output = Model::runModel(input); // call super
///     ... postprocess output
///     output = output * cppflow::tensor({-1}); // invert back, etc
///     return output; // done
/// }
///
class Model {

public:

	Model() = default;
	Model(const std::string & modelPath);
	virtual ~Model();

	/// load a SavedModel directory relative to bin/data
	/// directories for SavedModels include assets, variables, and a .pb file
	/// returns true on success
	virtual bool load(const std::string & modelPath);

	// set in and output names or reset to default names (call without args)
	// use the CLI tool "saved_model_cli" to inspect the SavedModel e.g.
	// saved_model_cli show --dir path/to/model/ --tag_set serve 
	// 						--signature_def serving_default
	virtual void setup(
		const std::vector<std::string> & inputNames,
		const std::vector<std::string> & outputNames);

	/// clear model
	virtual void clear();
	
	/// run model on input, blocks until returning output
	/// implement in a subclass to add custom pre / post processing
	virtual cppflow::tensor runModel(const cppflow::tensor & input) const;

	/// run model on mutiple inputs and outputs, blocks until returning output
	/// * inputs need to be given in the same order as defined in model settings
	/// * outputs are returned in the same manner as defined in model settings
	/// implement in a subclass to add custom pre / post processing
	virtual std::vector<cppflow::tensor>
	runMultiModel(const std::vector<cppflow::tensor> & inputs) const;

	/// returns true if model is loaded
	virtual bool isLoaded();

	/// print model operations ie. the signature
	virtual void printOperations();

protected:

	std::string modelPath_ = "";
	std::vector<std::string> inputNames_  = {{"serving_default_input_1"}};
	std::vector<std::string> outputNames_ = {{"StatefulPartitionedCall"}};
	cppflow::model * model_ = nullptr;
};

}; // end namespace ofxTF2
