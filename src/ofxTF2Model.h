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
	ofxTF2Tensor run(const ofxTF2Tensor & tensor);

protected:

	ofxTF2ModelSettings settings;
	std::string modelPath_;
	cppflow::model * model_ = nullptr;
};
