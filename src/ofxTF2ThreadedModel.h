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

#include "ofxTF2Model.h"
#include "ofThread.h"

/// \class ThreadedModel
/// \brief Model which processes input to output on a background thread
///
/// basic usage example:
///
/// class ofApp : public ofBaseApp {
/// 	public:
/// 		//...
/// 		void setup();
/// 		void update();
/// 	private:
/// 		ThreadedModel model;
/// 		//...
/// };
///
/// void ofApp::setup() {
/// 	// load the model and start the thread
/// 	model.load("path/to/modeldir");
/// 	model.startThread();
/// 	//...
/// }
///
/// void ofApp::update() {
/// 	// check if the model is ready to receive a new input tensor
/// 	if(model.readyForInput()) {
/// 		// create a tensor and feed it to the network
/// 		cppflow::tensor input = returnTensor();
/// 		model.update(input);
/// 	}
/// 	//...
/// 	// check if the model is done with computation
/// 	if(model.isOutputNew()) {
/// 		// receive and process the output of the network
/// 		cppflow::tensor output = model.getOutput();
/// 		doSomething(output);
/// 	}
/// 	//...
/// }
/// 
///
/// note: the thread is stopped & joined automatically in the destructor, if
///       you need to control this manually in ofApp::exit() or otherwise use:
///
///     model.waitForThread(bool callStopThread, long milliseconds);
///
///
/// \brief specify pre and postprocessing in a derived class by overriding the
///			runModel function. The way you handle the class stays the same as
///			described above.
///
/// class MyThreadedModel : public ThreadedModel {
/// 	public:
///     cppflow::tensor runModel(const cppflow::tensor & input) const override {
///			// preprocess: add one
///			auto modifiedInput = cppflow::add(input, {1});
/// 		// call to super runModel! Keep the call but change the input.
/// 		auto output = Model::runModel(modifiedInput);
/// 		// postprocess: multiply by minus one
/// 		return cppflow::mul(output, {-1});
/// 	}
/// };

namespace ofxTF2 {

class ThreadedModel : public Model, public ofThread {

public:

	/// stop and wait for thread to exit on delete
	virtual ~ThreadedModel();

	/// thread-safe call to Model::load()
	bool load(const std::string & modelPath) override;

	/// thread-safe call to Model::clear()
	void clear() override;

	/// returns true if the model is idle and ready for new input
	bool readyForInput();

	/// updates the model's current input
	/// returns true if the input was set or false if the model was busy
	bool update(cppflow::tensor input);

	/// returns true if output is ready
	/// note: subsequent calls will return false until the model has processed
	///       new input
	bool isOutputNew();

	/// get the current processed output
	/// note: subsequent calls may return different output if the model has
	///       processed new input
	cppflow::tensor getOutput();

	/// set thread idle sleep time in ms, default 100
	/// lower values will check for input more often at the expense of cpu time
	/// note: do not call this while the thread is running
	void setIdleTime(unsigned int ms);

	// override the runModel function so derived classes can redefine it
	virtual cppflow::tensor runModel(const cppflow::tensor & input) const override;

protected:

	/// thread run function, do not call this directly
	void threadedFunction();

	/// idle sleep time in ms
	/// note: not mutex protected, only set when the thread is not running
	unsigned int idleMS_ = 16;

	cppflow::tensor input_;  //< input to be processed, mutex protected
	cppflow::tensor output_; //< processed output, mutex protected
	bool newInput_ = false;  //< is the input new? mutex protected
	bool newOutput_ = false; //< is the output new? mutex protected
};

}; // end namespace ofxTF2
