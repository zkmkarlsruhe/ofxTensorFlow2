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

namespace ofxTF2 {

	typedef int64_t shape_t;
	typedef std::vector<shape_t> shapeVector;

	/// convert vector to string with a similar format to cppflow
	/// ex: {1, 2, 3} -> "[1, 2, 3, 4]"
	template<typename T>
	std::string vectorToString(const std::vector<T> & vec);

	/// map all values of a tensor from one range to another
	cppflow::tensor mapTensorValues(const cppflow::tensor & inputTensor,
		float inputMin, float inputMax, float outputMin, float outputMax);

	/// returns the shape of a tensor as std::vector<int64_t>
	/// as we found that a shape must be retrieved as int32_t
	/// we convert from int32_t to stick with the convention of int64_t
	shapeVector getTensorShape(const cppflow::tensor & tensor);

	/// returns true if number and size of dimensions are the same
	bool isSameShape(const shapeVector & lhs, 
		const shapeVector & rhs);

	/// converts a std::vector to a tensor
	/// type of resulting tensor is the TF equivalent to type of srcVector 
	/// creates a flat tensor if no shape provided
	/// only supports HWC output format (height, width, channel)
	/// does not convert tensor to a batch
	template <typename T>
	cppflow::tensor vectorToTensor(const std::vector<T> & srcVector, 
		const shapeVector & shape = shapeVector{0});

	/// converts ofPixels to a tensor
	/// type of resulting tensor is the TF equivalent to type of pixels
	/// e.g. TF_FLOAT for ofFloatPixels and TF_UINT8 for ofPixels 
	/// only supports HWC output format (height, width, channel)
	/// does not convert tensor to a batch
	/// returns tensor(-1) if not successful
	template <typename T>
	cppflow::tensor pixelsToTensor(const ofPixels_<T> & pixels);

	/// converts ofImage to a tensor
	/// type of resulting tensor is the TF equivalent to type of image
	/// e.g. TF_FLOAT for ofFloatImage and TF_UINT8 for ofImage 
	/// only supports HWC output format (height, width, channel)
	/// does not convert tensor to a batch
	/// returns tensor(-1) if not successful
	template <typename T>
	cppflow::tensor imageToTensor(const ofImage_<T> & image);

	/// converts a tensor to std::vector
	/// casts data type of tensor to data type of vector if they mismatch
	/// only supports HWC output format (height, width, channel)
	/// return true if successful
	template <typename T>
	bool tensorToVector(const cppflow::tensor & srcTensor, std::vector<T> & dstVector);

	/// converts a tensor to any ofPixels_<T>
	/// casts data type of tensor to data type of pixels if they mismatch
	/// only supports HWC input format (height, width, channel)
	/// input may have batch size of one (1, H,W,C) or no batch (H,W,C)
	/// return true if successful
	template <typename T>
	bool tensorToPixels(const cppflow::tensor & srcTensor, ofPixels_<T> & pixels);
	
	/// converts a tensor to any ofImage_<T>
	/// casts data type of tensor to data type of image if they mismatch
	/// only supports HWC input format (height, width, channel)
	/// input may have batch size of one (1, H,W,C) or no batch (H,W,C)
	/// return true if successful
	template <typename T>
	bool tensorToImage(const cppflow::tensor & srcTensor, ofImage_<T> & image);

}; // end namespace ofxTF2

namespace ofxTF2 {

	// ==== template implementations ====

	template<typename T>
	std::string vectorToString(const std::vector<T> & vec) {
		std::string s("[");
		for (std::size_t i = 0; i < vec.size(); i++) {
			s.append(std::to_string(vec[i]));
			if(i != vec.size() - 1) {
				s.append(", ");
			}
		}
		s.append("]");
		return s;
	}

	template <typename T>
	cppflow::tensor vectorToTensor(const std::vector<T> & srcVector,
	                               const shapeVector & shape) {
		auto shape_ = shape; 
		// by default and if shape is (0) create a flat vector
		if(shape == shapeVector{0}) {
			shape_ = {(size_t) srcVector.size()};
		}
		return cppflow::tensor(srcVector, shape_);
	}

	template <typename T>
	cppflow::tensor imageToTensor(const ofImage_<T> & image) {
		return pixelsToTensor(image.getPixels());
	}

	template <typename T>
	cppflow::tensor pixelsToTensor(const ofPixels_<T> & pixels) {
		const shape_t w = pixels.getWidth();
		const shape_t h = pixels.getHeight();
		shape_t c;
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
				ofLogError("ofxTensorFlow2") << "pixelsToTensor unknown image type: "
					<< std::to_string(pixels.getImageType());
				return cppflow::tensor(-1);
		}
		return cppflow::tensor(std::vector<T>(pixels.begin(), pixels.end()), {h, w, c});
	}

	template <typename T>
	bool tensorToVector(const cppflow::tensor & srcTensor, std::vector<T> & dstVector) {
		try {
			// get in and output type as TF_DATA_TYPE
			auto inputType  = srcTensor.dtype();
			auto outputType = cppflow::deduce_tf_type<T>();
			// if necessary, cast srcTensor to the TF_Version of T before copying
			if (inputType != outputType) {
				cppflow::tensor tempCast = cppflow::cast(srcTensor, inputType, outputType);
				dstVector = tempCast.get_data<T>();
				ofLogVerbose("ofxTensorFlow2") << "tensorToVector"
					<< " cast tensor type " << cppflow::to_string(inputType)
					<< " to vector type "  << cppflow::to_string(outputType);
			}
			else {
				dstVector = srcTensor.get_data<T>();
			}
		}
		catch(const std::exception& e) {
			ofLogError("ofxTensorFlow2") << "tensorToVector copy without cast: "
				<< e.what();
			dstVector = srcTensor.get_data<T>();
			return false;
		}
		return true;
	}

	template <typename T>
	bool tensorToImage(const cppflow::tensor & srcTensor, ofImage_<T> & image) {
		return tensorToPixels(srcTensor, image.getPixels());
	}

	template <typename T>
	bool tensorToPixels(const cppflow::tensor & srcTensor, ofPixels_<T> & pixels) {
		// get tensor shape and check if convertible
		auto shape = getTensorShape(srcTensor);
		shape_t tensor_w;
		shape_t tensor_h;
		shape_t tensor_c;
		// tensor is not a batch
		if(shape.size() == 3) {
			tensor_w = shape[1];
			tensor_h = shape[0];
			tensor_c = shape[2];
		}
		// tensor is a batch
		else if(shape.size() == 4) {
			if(shape[0] != 1) {
				ofLogError("ofxTensorFlow2") << "tensorToPixels supports only batch sizes of 1";
				return false;
			}
			tensor_w = shape[2];
			tensor_h = shape[1];
			tensor_c = shape[3];
		}
		// tensor cannot be converted to pixels
		else {
			ofLogError("ofxTensorFlow2") << "tensorToPixels wrong number of channels. "
				<< std::to_string(shape.size()) << " dimensions given, "
				<< "but expected 3 or 4 (if batch size is 1)";
			return false;
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
			default:
				ofLogError("ofxTensorFlow2") << "tensorToPixels unknown pixels image type";
				return false;
		}
		// check if shapes matches
		shapeVector tensorShape = {tensor_w, tensor_h, tensor_c};
		shapeVector pixelsShape = {pixels_w, pixels_h, pixels_c};
		if(!isSameShape(tensorShape, pixelsShape)) {
			return false;
		}
		// copy to pixels
		std::vector<T> data;
		tensorToVector(srcTensor, data);
		std::copy(data.begin(), data.end(), pixels.begin());
		
		return true;
	}

	/// set TensorFlow log level using ofLogLevel enums
	/// FIXME: doesn't seem to work yet...
	void setLogLevel(ofLogLevel level);

}; // end namespace ofxTF2
