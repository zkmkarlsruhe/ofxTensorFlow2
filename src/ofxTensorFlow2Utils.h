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

	/// convert vector to string with a similar format to cppflow
	/// ex: {1, 2, 3} -> "[1, 2, 3, 4]"
	template<typename T>
	static std::string vectorToString(const std::vector<T> & vec);

	/// set TensorFlow log level using ofLogLevel enums
	/// FIXME: doesn't seem to work yet, at least on macOS...
	static void setLogLevel(ofLogLevel level);
};

namespace cppflow {

// currently assumes tenspr range float -1 to 1
tensor image_to_tensor(const ofImage & image);

// currently assumes tenspr range float -1 to 1
void tensor_to_image(const tensor & tensor, ofImage & image);

// currently assumes tenspr range float -1 to 1
tensor pixels_to_tensor(const ofPixels & pixels);

// currently assumes tenspr range float -1 to 1
void tensor_to_pixels(const tensor & tensor, ofPixels & pixels);

tensor map_values(const tensor & inputTensor, float inputMin,
	float inputMax, float outputMin, float outputMax);

};

// ==== template implementations ====

template<typename T>
std::string ofxTensorFlow2::vectorToString(const std::vector<T> & vec) {
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
