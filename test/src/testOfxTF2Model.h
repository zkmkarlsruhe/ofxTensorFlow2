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
#include "ofxTensorFlow2.h"

int testOfxTF2Model(const cppflow::tensor & input, const cppflow::tensor & output){

	// ====== ofxTF2Model ====== //
	ofLog() << "============= Start testing ofxTF2Model =============";

	// load the ofxTF2Model
	ofxTF2Model simpleModel(ofToDataPath("model"));
	ofLog() << "testOfxTF2Model: successfully loaded model";

	// reload
	simpleModel.load(ofToDataPath("model"));
	ofLog() << "testOfxTF2Model: successfully reloaded model";

	// run from cppflow::input
	auto simpleModelOutput = simpleModel.run(input);
	ofLog() << "testOfxTF2Model: successfully ran model on cppflow tensor";

	// run from converted input
	// convert input to an ofxTF2Tensor
	ofxTF2Tensor inputConverted(input);
	ofLog() << "testOfxTF2Model: successfully converted cppflow tensor";
	auto simpleModelOutput2 = simpleModel.run(inputConverted);
	ofLog() << "testOfxTF2Model: successfully ran model on ofxTF2Tensor";

	// compare to output
	ofLog() << "testOfxTF2Model: cppflow::model(cppflow::tensor): "
		<< vectorToString(output.get_data<int>());
	ofLog() << "testOfxTF2Model: ofxTF2Model(cppflow::tensor): "
		<< vectorToString(simpleModelOutput.getVector<int>());
	ofLog() << "testOfxTF2Model: ofxTF2Model(ofxTF2Tensor): "
		<< vectorToString(simpleModelOutput2.getVector<int>());
		
	return 0;
}
