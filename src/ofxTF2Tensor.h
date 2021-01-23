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
#include "ofImage.h"
#include "ofLog.h"

// TODO: NEED static method that returns the data type
// TODO: test image and pixel inputs
// TODO: move shapeToString to ofxTensorFlow2Utils.h
// TODO: write and test image and pixel outputs

// a shape is represented as a vector of shape_t
typedef int32_t shape_t;

/// \class ofxTF2Tensor
/// \brief a simple wrapper for cppflow::tensor
/// note: getData<T> (and cppflow::tensor.get_data<T>) does not convert the value
class ofxTF2Tensor{

public:

    /// \section Constructors

    // forwarding to cppflow constructors
    template <typename T>
	ofxTF2Tensor(const T& value);

    template <typename T>
	ofxTF2Tensor(const std::vector<T> & values);

    template <typename T>
	ofxTF2Tensor(const std::vector<T> & values,
				 const std::vector<int64_t>& shape);

	ofxTF2Tensor(const cppflow::tensor & tensor);

    // constructors for openframework interaction
    template <typename T>
	ofxTF2Tensor(const ofPixels & pixels);

	ofxTF2Tensor(const ofImage & img);

    /// \section Operators

    /// implicit cast to cppflow::tensor
    /// especially useful for using cppflow ops with ofxTF2Tensor
	operator cppflow::tensor & ();

    /// check if tensors are comparable
	bool operator == (const cppflow::tensor & rhs) const;
    bool operator == (const ofxTF2Tensor & rhs) const;

	/// std::ostream print operator
	friend std::ostream & operator << (std::ostream & os, const ofxTF2Tensor & tensor);

	/// \section Comparison

    /// check if both tensors share the same values
    template <typename T>
	bool equals(const cppflow::tensor & rhs) const;
    template <typename T>
    bool equals (const ofxTF2Tensor & rhs) const;

	/// \section Data Access

	std::vector<shape_t> getShape() const;

	const cppflow::tensor & getTensor() const;

    /// this function will not convert the values. Make sure to use the correct
    /// type. That is, when using int as template parameter on a float tensor
    /// values will not get truncated but the bytes will be interpreted as int.
    template <typename T>
	std::vector<T> getVector() const;

protected:

    cppflow::tensor tensor_;
};

// ==== template implementations ====

template <typename T>
ofxTF2Tensor::ofxTF2Tensor(const T& value)
	: tensor_(value) {}

template <typename T>
ofxTF2Tensor::ofxTF2Tensor(const std::vector<T> & values)
	: tensor_(values) {}

template <typename T>
ofxTF2Tensor::ofxTF2Tensor(const std::vector<T> & values,
						   const std::vector<int64_t>& shape)
	: tensor_(values, shape) {}

template <typename T>
ofxTF2Tensor::ofxTF2Tensor(const ofPixels & pixels) : tensor_(
		std::vector<T>(pixels.begin(), pixels.end()),
		{pixels.getWidth(), pixels.getHeight(), pixels.getNumChannels()})
		{}

template <typename T>
bool ofxTF2Tensor::equals(const cppflow::tensor & rhs) const {
	if (!( *this == rhs )){
		ofLogWarning() << "ofxTF2Tensor: tensors not comparable";
		return false;
	}
	if (tensor_.get_data<T>() != rhs.get_data<T>()) {
		ofLogWarning() << "ofxTF2Tensor: value mismatch";
		return false;
	}
	return true;
}

template <typename T>
bool ofxTF2Tensor::equals (const ofxTF2Tensor & rhs) const {
    return this->equals<T>(rhs.getTensor());
}

template <typename T>
std::vector<T> ofxTF2Tensor::getVector() const {
	return tensor_.get_data<T>();
}
