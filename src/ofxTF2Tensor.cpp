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

#include "ofxTF2Tensor.h"
#include "ofxTensorFlow2Utils.h"

// ==== constructors ====

ofxTF2Tensor::ofxTF2Tensor(const cppflow::tensor & tensor) :
	tensor_(tensor) {}

ofxTF2Tensor::ofxTF2Tensor(const ofImage & img) :
	ofxTF2Tensor(img.getPixels()) {}

// ==== operators ====

ofxTF2Tensor::operator cppflow::tensor & (){
	return tensor_;
}

bool ofxTF2Tensor::operator == (const cppflow::tensor & rhs) const{

	// check if shapes are the same
	auto lhsShape = tensor_.shape().get_data<shape_t>();
	auto rhsShape = rhs.shape().get_data<shape_t>();
	if (lhsShape != rhsShape){
		ofLogWarning() << "ofxTF2Tensor: shape mismatch:"
		               << " shape(lhs): " << vectorToString(lhsShape)
		               << " shape(rhs): " << vectorToString(rhsShape);
		return false;
	}

	// check if the data types are the same
	if (tensor_.dtype() != rhs.dtype()){
		ofLogWarning() << "ofxTF2Tensor: dtype mismatch";
		return false;
	}

	return true;
}

bool ofxTF2Tensor::operator == (const ofxTF2Tensor & rhs) const {
	return (*this) == rhs.getTensor();
}

std::ostream & operator << (std::ostream & os, const ofxTF2Tensor & tensor){
	return os << tensor.tensor_;
}

// ==== data access ====

void ofxTF2Tensor::fill(const cppflow::tensor& dims,
						const cppflow::tensor& value,
						cppflow::datatype index_type) {
	tensor_ = cppflow::fill(dims, value, index_type);
}

std::vector<shape_t> ofxTF2Tensor::getShape() const{
	return tensor_.shape().get_data<shape_t>();
}

const cppflow::tensor & ofxTF2Tensor::getTensor() const{
	return tensor_;
}
