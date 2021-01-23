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

int testOperators(const cppflow::tensor & input){

    ofLog() << "====== Start testing Operators ======";
	// ofxTF2Tensor from cppflow::tensor
	ofxTF2Tensor fromCppflowTensor (input);

    // check comparision operator
	auto cppflowTensorOtherShape = cppflow::fill({2, 1, 2, 3}, 0.9f);
	auto cppflowTensorOtherType = cppflow::fill({1, 2, 2, 3}, 1);
    ofLog() << "testOfxTF2Tensor: ofxTensor(input) == input: " 
			<< (fromCppflowTensor == input);
    ofLog() << "testOfxTF2Tensor: ofxTensor(input) == cppflow::tensor(othershape): " 
			<< (fromCppflowTensor == cppflowTensorOtherShape);
    ofLog() << "testOfxTF2Tensor: ofxTensor(input) == cppflow::tensor(othertype): " 
			<< (fromCppflowTensor == cppflowTensorOtherType);

    // check value 
    bool isEqualSelf = fromCppflowTensor.equals<float>(input);
    ofLog() << "testOfxTF2Tensor: ofxTensor(input) equal to input: " 
			<< isEqualSelf;
	auto cppflowTensorOtherValue = cppflow::fill({1, 2, 2, 3}, 0.99f);
    bool isEqualOtherValue = fromCppflowTensor.equals<float>(cppflowTensorOtherValue);
    ofLog() << "testOfxTF2Tensor: ofxTensor(input) cppflow::tensor(otherValue): " 
			<< isEqualOtherValue;

    // todo ostream operator not working
    // ofLog() << fromCppflowTensor << std::endl;
    // ofLog() << fromCppflowTensor.getTensor();

	
	return 1;
}

int testConstructorsFloatVector(const std::vector<float> & input){

	std::vector<int64_t> shape(1);
	shape[0] = input.size();

	ofxTF2Tensor fromFloatVector (input, shape);
	auto vecFromFloatVector = fromFloatVector.getVector<float>();
	
	ofLog() << "testOfxTF2Tensor: ofxTensor(input) == floatVector: "
			<< (vecFromFloatVector == input);
    ofLog() << vectorToString(input);
    ofLog() << vectorToString(vecFromFloatVector);

    return 1;
}

int testConstructors(){

    ofLog() << "====== Start testing Operators ======";

    std::vector<float> floatVector
		{0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
	testConstructorsFloatVector(floatVector);

	// TODO: ofxTF2Tensor from ofImage
	// TODO: ofxTF2Tensor from ofPixels

	// TODO: ofImage from ofxTF2Tensor  
	// TODO: ofPixels from ofxTF2Tensor

	return 1;
}


int testOfxTF2Tensor(const cppflow::tensor & input){

	// ====== ofxTF2Tensor ====== //
    ofLog() << "============= Start testing ofxTF2Tensor =============";
	testOperators(input);
	testConstructors();

	return 1;
}
