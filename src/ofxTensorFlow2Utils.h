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

/// static util class
class ofxTensorFlow2 {

public:
	/// set TensorFlow log level using ofLogLevel enums
	/// FIXME: doesn't seem to work yet, at least on macOS...
	static void setLogLevel(ofLogLevel level);
};


namespace ofxTF2{

	using namespace cppflow;

	// shape must be retrieved using uint32_t, however tensor() uses uint64_t
	typedef uint32_t shape_t;

	/// convert vector to string with a similar format to cppflow
	/// ex: {1, 2, 3} -> "[1, 2, 3, 4]"
	template<typename T>
	std::string vectorToString(const std::vector<T> & vec);

	// map all values of a tensor from one range to another
	tensor mapTensorValues(const tensor & inputTensor, float inputMin,
		float inputMax, float outputMin, float outputMax);

	// returns the shape of a tensor
	std::vector<shape_t> getTensorShape(const tensor & tensor);

	// returns true if number and size of dimensions are the same
	bool isSameShape (const std::vector<shape_t> & lhs, 
		const std::vector<shape_t> & rhs);

	// converts a std::vector to a tensor
	// expects shape in HWC format (height, width, channel)
	// creates a flat tensor if no shape provided 
	// does not convert tensor to a batch
	// return 0 if successful
	template <typename T>
	int vectorToTensor(const std::vector<T> & vector, tensor & dstTensor, 
		std::vector<shape_t> shape);

	// converts a std::vector to a tensor
	// only supports HWC output format (height, width, channel)
	// return 0 if successful
	template <typename T>
	int tensorToVector(const tensor & srcTensor, std::vector<T> & vector);

	// converts ofImage to a tensor
	// only supports HWC output format (height, width, channel)
	// does not convert tensor to a batch
	// return 0 if successful
	template <typename T>
	int imageToTensor(const ofImage & image, tensor & dstTensor);

	// converts a tensor to ofImage
	// only supports HWC input format (height, width, channel)
	// input may have batch size of one (1, H,W,C) or no batch (H,W,C)
	// return 0 if successful
	template <typename T>
	int tensorToImage(const tensor & srcTensor, ofImage & image);

	// converts ofPixels to a tensor
	// only supports HWC output format (height, width, channel)
	// does not convert tensor to a batch
	// return 0 if successful
	template <typename T>
	int pixelsToTensor(const ofPixels & pixels, tensor & dstTensor);

	// converts a tensor to ofPixels
	// only supports HWC input format (height, width, channel)
	// input may have batch size of one (1, H,W,C) or no batch (H,W,C)
	// return 0 if successful
	template <typename T>
	int tensorToPixels(const tensor & srcTensor, ofPixels & pixels);


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
	int vectorToTensor(const std::vector<T> & srcVector, tensor & dstTensor, 
			std::vector<uint64_t> shape = std::vector<uint64_t>{0}) {
		if (shape == std::vector<uint64_t>{0}) // create a flat vector
			dstTensor = tensor(srcVector);
		else
			dstTensor = tensor(srcVector, shape);
		return 0;
	}

	template <typename T>
	int tensorToVector(const tensor & srcTensor, std::vector<T> & dstVector){
		dstVector = srcTensor.get_data<T>();
		return 0;
	}

	template <typename T>
	int imageToTensor(const ofImage & image, tensor & dstTensor) {
		return pixelsToTensor<T>(image.getPixels(), dstTensor);
	}

	template <typename T>
	int tensorToImage(const tensor & srcTensor, ofImage & image) {
		return tensorToPixels<T>(srcTensor, image.getPixels());
	}

	template <typename T>
	int pixelsToTensor(const ofPixels & pixels, tensor & dstTensor) {
		const shape_t w = pixels.getWidth();
		const shape_t h = pixels.getHeight();
		std::vector<int64_t> shape;
		switch(pixels.getImageType()) {
			case OF_IMAGE_GRAYSCALE:
				shape = {h, w, 1};
				break;
			case OF_IMAGE_COLOR:
				shape = {h, w, 3};
				break;
			case OF_IMAGE_COLOR_ALPHA:
				shape = {h, w, 4};
				break;
			case OF_IMAGE_UNDEFINED:
				ofLogError() << "pixels_to_tensor: image type undefined: " 
							<< std::to_string(pixels.getImageType());
				return -1;
			default:
				ofLogError() << "pixels_to_tensor: image type not implemented";
				return -1;
		}
		dstTensor = tensor(std::vector<T>(pixels.begin(), pixels.end()), shape);
		return 0;
	}


	template <typename T>
	int tensorToPixels(const tensor & srcTensor, ofPixels & pixels) {

		// get tensor shape and check if convertible
		auto shape = getTensorShape(srcTensor);
		shape_t tensor_w;
		shape_t tensor_h;
		shape_t tensor_c;
		// tensor is not a batch
		if (shape.size() == 3){
			tensor_w = shape[1];
			tensor_h = shape[0];
			tensor_c = shape[2];
		}
		// tensor is a batch
		else if (shape.size() == 4){
			if (shape[0] != 1){
				ofLogError() << "tensor_to_pixels: supports only batch_sizes of 1";
				return -1;
			}
			tensor_w = shape[2];
			tensor_h = shape[1];
			tensor_c = shape[3];
		}
		// tensor cannot be converted to pixels
		else
		{
			ofLogError() <<  "tensor_to_pixels: wrong number of channels. "
				<< std::to_string(shape.size())
				<< " dimensions given, but expected 3 or 4 (if batch size is 1)";
			return -1;
		}
		
		// get ofPixels shape
		shape_t pixels_w = pixels.getWidth();
		shape_t pixels_h = pixels.getHeight();
		shape_t pixels_c;
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
				ofLogError() << "tensor_to_pixels: pixels image type undefined";
				return -1;
			default:
				ofLogError() << "tensor_to_pixels: pixels image type not implemented";
				return -1;
		}

		// check if shapes matches
		if (!isSameShape( {tensor_w, tensor_h, tensor_c},
							{pixels_w, pixels_h, pixels_c}) ) {
			return -1;
		}

		// copy to pixels
		auto data = srcTensor.get_data<T>();
		std::copy(data.begin(), data.end(), pixels.begin());
		
		return 0;
	}

}; // end namespace ofxTF2
