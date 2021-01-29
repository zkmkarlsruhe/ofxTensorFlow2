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

	
typedef std::vector<int32_t> shape_t;

namespace ofxTF2{

	/// convert vector to string with a similar format to cppflow
	/// ex: {1, 2, 3} -> "[1, 2, 3, 4]"
	template<typename T>
	std::string vectorToString(const std::vector<T> & vec);

	// map all values of a tensor from one range to another
	cppflow::tensor mapTensorValues(const cppflow::tensor & inputTensor, float inputMin,
		float inputMax, float outputMin, float outputMax);

	// returns the shape of a tensor
	shape_t getTensorShape(const cppflow::tensor & tensor);

	// returns true if number and size of dimensions are the same
	bool isSameShape(const shape_t & lhs, 
		const shape_t & rhs);

	// converts a std::vector to a tensor
	// expects shape in HWC format (height, width, channel)
	// creates a flat tensor if no shape provided 
	// does not convert tensor to a batch
	// returns tensor(-1) if not successful
	template <typename T>
	cppflow::tensor vectorToTensor(const std::vector<T> & srcVector, 
		shape_t shape = shape_t{0});

	// converts ofPixels to a tensor
	// only supports HWC output format (height, width, channel)
	// does not convert tensor to a batch
	// returns tensor(-1) if not successful
	template <typename T>
	cppflow::tensor pixelsToTensor(const ofPixels & pixels);

	// converts ofImage to a tensor
	// only supports HWC output format (height, width, channel)
	// does not convert tensor to a batch
	// returns tensor(-1) if not successful
	template <typename T>
	cppflow::tensor imageToTensor(const ofImage & image);

	// converts a std::vector to a tensor
	// only supports HWC output format (height, width, channel)
	// return true if successful
	template <typename T>
	bool tensorToVector(const cppflow::tensor & srcTensor, std::vector<T> & dstVector);

	// converts a tensor to ofPixels
	// only supports HWC input format (height, width, channel)
	// input may have batch size of one (1, H,W,C) or no batch (H,W,C)
	// return true if successful
	template <typename T>
	bool tensorToPixels(const cppflow::tensor & srcTensor, ofPixels & pixels);
	
	// converts a tensor to ofImage
	// only supports HWC input format (height, width, channel)
	// input may have batch size of one (1, H,W,C) or no batch (H,W,C)
	// return true if successful
	template <typename T>
	bool tensorToImage(const cppflow::tensor & srcTensor, ofImage & image);

}; // end namespace ofxTF2

namespace ofxTF2{

	// ==== template implementations ====

	template<typename T>
	std::string vectorToString(const std::vector<T> & vec) {
		std::string s("[");
		for (int i = 0; i < vec.size(); i++) {
			s.append(std::to_string(vec[i]));
			if (i != vec.size() - 1) {
				s.append(", ");
			}
		}
		s.append("]");
		return s;
	}

	template <typename T>
	cppflow::tensor vectorToTensor(const std::vector<T> & srcVector, 
		shape_t shape) {
		// if shape is (0) create a flat vector
		if (shape == shape_t{0})
			return tensor(srcVector);
		else
			return tensor(srcVector, shape);;
	}

	template <typename T>
	cppflow::tensor imageToTensor(const ofImage & image) {
		return pixelsToTensor<T>(image.getPixels());
	}

	template <typename T>
	cppflow::tensor pixelsToTensor(const ofPixels & pixels) {
		const int32_t w = pixels.getWidth();
		const int32_t h = pixels.getHeight();
		int32_t c;
		switch(pixels.getImageType()) {
			case OF_IMAGE_GRAYSCALE:
				c = 1;
				break;
			case OF_IMAGE_COLOR:
				c = 3;
				break;
			case OF_IMAGE_COLOR_ALPHA:
				c = 4;
				break;
			case OF_IMAGE_UNDEFINED:
			default:
				ofLogError() << "ofxTensorFlow2: pixelsToTensor unknown image type: "
							<< std::to_string(pixels.getImageType());
				return cppflow::tensor(-1);
		}
		return cppflow::tensor(std::vector<T>(pixels.begin(), pixels.end()), {h, w, c});
	}

	template <typename T>
	bool tensorToVector(const cppflow::tensor & srcTensor, std::vector<T> & dstVector){
		dstVector = srcTensor.get_data<T>();
		return true;
	}

	template <typename T>
	bool tensorToImage(const cppflow::tensor & srcTensor, ofImage & image) {
		return tensorToPixels<T>(srcTensor, image.getPixels());
	}

	template <typename T>
	bool tensorToPixels(const cppflow::tensor & srcTensor, ofPixels & pixels) {
		// get tensor shape and check if convertible
		auto shape = getTensorShape(srcTensor);
		int32_t tensor_w;
		int32_t tensor_h;
		int32_t tensor_c;
		// tensor is not a batch
		if (shape.size() == 3){
			tensor_w = shape[1];
			tensor_h = shape[0];
			tensor_c = shape[2];
		}
		// tensor is a batch
		else if (shape.size() == 4){
			if (shape[0] != 1){
				ofLogError() << "ofxTensorFlow2: tensorToPixels supports only batch sizes of 1";
				return false;
			}
			tensor_w = shape[2];
			tensor_h = shape[1];
			tensor_c = shape[3];
		}
		// tensor cannot be converted to pixels
		else
		{
			ofLogError() << "ofxTensorFlow2: tensorToPixels wrong number of channels. "
				<< std::to_string(shape.size())
				<< " dimensions given, but expected 3 or 4 (if batch size is 1)";
			return false;
		}
		// get ofPixels shape
		int32_t pixels_w = pixels.getWidth();
		int32_t pixels_h = pixels.getHeight();
		int32_t pixels_c;
		switch(pixels.getImageType()) {
			case OF_IMAGE_GRAYSCALE:
				pixels_c = 1;
				break;
			case OF_IMAGE_COLOR:
				pixels_c = 3;
				break;
			case OF_IMAGE_COLOR_ALPHA:
				pixels_c = 4;
				break;
			case OF_IMAGE_UNDEFINED:
			default:
				ofLogError() << "ofxTensorFlow2: tensorToPixels unknown pixels image type";
				return false;
		}
		// check if shapes matches
		if (!isSameShape( {tensor_w, tensor_h, tensor_c},
							{pixels_w, pixels_h, pixels_c}) ) {
			return false;
		}
		// copy to pixels
		auto data = srcTensor.get_data<T>();
		std::copy(data.begin(), data.end(), pixels.begin());
		
		return true;
	}

	/// set TensorFlow log level using ofLogLevel enums
	/// FIXME: doesn't seem to work yet, at least on macOS...
	void setLogLevel(ofLogLevel level);

}; // end namespace ofxTF2
