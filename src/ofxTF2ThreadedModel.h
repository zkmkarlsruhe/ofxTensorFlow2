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

/// \class ofxTF2ThreadedModel
/// \brief ofxTF2Model which processes input to output on a background thread
///
/// basic usage example:
///
///     class ofApp : public ofBaseApp {
///     public:
///     ...
///         ofxTF2ThreadedModel model;
///     };
///
///     void ofApp::setup() {
///         ...
///         model.load("path/to/modeldir");
///         model.startThread();
///     }
///
///     void ofApp.cpp::update() {
///         if(model.readyForInput()) {
///             cppflow::tensor input = ...
///             ... prepare input
///             model.update(input);
///         }
///         ...
///         if(model.isOutputNew()) {
///             cppflow::tensor output = model.getOutput();
///             ... process output
///         }
///     }
///
/// note: the thread is stopped & joined automatically in the destructor, if
///       you need to control this manually in ofApp::exit() or otherwise use:
///
///     model.waitForThread(bool callStopThread, long milliseconds);
///
class ofxTF2ThreadedModel : public ofxTF2Model, public ofThread {

public:

	/// stop and wait for thread to exit on delete
	virtual ~ofxTF2ThreadedModel();

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

	// locked call to clear()
	void clearSafely();

	// locked call to load()
	void loadSafely(const std::string & modelPath);

protected:

	/// thread run function, do not call this directly
	void threadedFunction();

	/// idle sleep time in ms
	/// TODO: should this be shorter? ie. 16 ms ~= 60 fps?
	/// note: not mutex protected, only set when the thread is not running
	unsigned int idleMS_ = 100;

	cppflow::tensor input_;  //< input to be processed, mutex protected
	cppflow::tensor output_; //< processed output, mutex protected
	bool newInput_ = false;  //< is the input new? mutex protected
	bool newOutput_ = false; //< is the output new? mutex protected
};
