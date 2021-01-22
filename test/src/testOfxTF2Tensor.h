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

int testOfxTF2Tensor(){

	// ====== ofxTF2Tensor ====== //
    ofLog() << "============= Start testing ofxTF2Tensor =============";

	// ofxTF2Tensor from cppflow::tensor
	auto cppflowTensor = cppflow::fill({1, 2, 2, 3}, 2);
	ofxTF2Tensor fromCppflowTensor (cppflowTensor);

    // check comparision operator
	auto cppflowTensorOtherShape = cppflow::fill({2, 1, 2, 3}, 0.9f);
	auto cppflowTensorOtherType = cppflow::fill({1, 2, 2, 3}, 1);
    ofLog() << "testOfxTF2Tensor: is same self: " << (fromCppflowTensor == cppflowTensor);
    ofLog() << "testOfxTF2Tensor: is same shape: " << (fromCppflowTensor == cppflowTensorOtherShape);
    ofLog() << "testOfxTF2Tensor: is same type: " << (fromCppflowTensor == cppflowTensorOtherType);

    // check value 
	auto cppflowTensorOtherValue = cppflow::fill({1, 2, 2, 3}, 0.99f);
    bool isEqualOtherValue = fromCppflowTensor.equals<float>(cppflowTensorOtherValue);
    bool isEqualSelf = fromCppflowTensor.equals<float>(cppflowTensor);
    ofLog() << "testOfxTF2Tensor: equal to other value: " << isEqualOtherValue;
    ofLog() << "testOfxTF2Tensor: equal to self: " << isEqualSelf;

    // todo ostream operator not working
    ofLog() << fromCppflowTensor << std::endl;
    // ofLog() << fromCppflowTensor.getTensor();

	// todo ofxTF2Tensor from vector
	// todo ofxTF2Tensor from ofImage
	// todo ofxTF2Tensor from ofPixels

	// vector from ofxTF2Tensor
	auto vecFromofxTF2Tensor = fromCppflowTensor.getVector<int>();

	// todo vector from ofxTF2Tensor
	// todo ofImage from ofxTF2Tensor  
	// todo ofPixels from ofxTF2Tensor

	// std::cout << "vecFromofxTF2Tensor size: " << vecFromofxTF2Tensor.size() << std::endl;
	// std::cout << "vecFromCppflowTensor size: " << input.get_data<float>().size() << std::endl;

    ofLog() << vectorToString(vecFromofxTF2Tensor);

    return 0;
}
