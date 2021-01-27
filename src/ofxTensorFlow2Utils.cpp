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

#include "ofxTensorFlow2Utils.h"
#include <cstdlib>
#include <sys/errno.h>
#include <float.h>
#include "ofConstants.h"

#ifdef TARGET_WIN32
/// Windows doesn't provide setenv(), use this _putenv_s() wrapper from:
/// https://stackoverflow.com/a/23616164/2146055
static setenv(const char *name, const char *value, int overwrite) {
	int errcode = 0;
	if(!overwrite) {
		size_t envsize = 0;
		errcode = getenv_s(&envsize, NULL, 0, name);
		if(errcode || envsize) return errcode;
	}
	return _putenv_s(name, value);
}
#endif

// the TensorFlow log level is set via the TF_CPP_MIN_LOG_LEVEL env variable,
// TF log const ints are defined in tensorflow/core/platform/default/logging.h
void ofxTensorFlow2::setLogLevel(ofLogLevel level) {
	int tfLogLevel;
	switch(level) {
		default:
		case OF_LOG_VERBOSE:
		case OF_LOG_NOTICE:
			tfLogLevel = 0; // INFO
			break;
		case OF_LOG_WARNING:
			tfLogLevel = 1; // WARNING
			break;
		case OF_LOG_ERROR:
			tfLogLevel = 2; // ERROR
		case OF_LOG_FATAL_ERROR:
			tfLogLevel = 3; // FATAL
		case OF_LOG_SILENT:
			tfLogLevel = 4; // NUM_SEVERITIES?
	}
	const std::string n = "TF_CPP_MIN_LOG_LEVEL";
	std::string v = ofToString(tfLogLevel);
	if(setenv(n.c_str(), v.c_str(), 1) == -1) {
		ofLogError() << "ofxTensorFlow2: unable to set TF_CPP_MIN_LOG_LEVEL: "
		             << errno << " " << strerror(errno);
	}
}

namespace cppflow {

tensor image_to_tensor(const ofImage & image) {
	return pixels_to_tensor(image.getPixels());
}

void tensor_to_image(const tensor & tensor, ofImage & image) {
	tensor_to_pixels(tensor, image.getPixels());
}

tensor pixels_to_tensor(const ofPixels & pixels) {
	const long long w = pixels.getWidth();
	const long long h = pixels.getHeight();
	switch(pixels.getImageType()) {
		case OF_IMAGE_GRAYSCALE:
			return tensor(
			std::vector<float>(pixels.begin(),
								pixels.end()),
							{h, w, 1});
		case OF_IMAGE_COLOR:
			return tensor(
			std::vector<float>(pixels.begin(),
								pixels.end()),
							{h, w, 3});
		case OF_IMAGE_COLOR_ALPHA:
			return tensor(
			std::vector<float>(pixels.begin(),
								pixels.end()),
							{h, w, 4});
		case OF_IMAGE_UNDEFINED:
			return tensor(-1);
		default:
			return tensor(-1);
	}
}

void tensor_to_pixels(const tensor & tensor, ofPixels & pixels) {
	auto data = tensor.get_data<float>();
	if(pixels.size() <= data.size()) {
	}
	switch(pixels.getImageType()) {
		case OF_IMAGE_GRAYSCALE:
			for(int i = 0; i < pixels.size(); i++){
				pixels[i] = data[i];
			}
			break;
		case OF_IMAGE_COLOR:
			for(int i = 0; i < pixels.size(); i+=3){
				pixels[i] = data[i];
				pixels[i+1] = data[i+1];
				pixels[i+2] = data[i+2];
			}
			break;
		case OF_IMAGE_COLOR_ALPHA:
			for(int i = 0; i < pixels.size(); i+=4){
				pixels[i] = data[i];
				pixels[i+1] = data[i+1];
				pixels[i+2] = data[i+2];
				pixels[i+3] = data[i+3];
			}
			break;
		case OF_IMAGE_UNDEFINED:
			ofLogError() << "tensor_to_pixels, pixels image type undefined";
			break; 
	}

}

tensor map_values(const tensor & inputTensor, float inputMin,
	float inputMax, float outputMin, float outputMax) {

	if (fabs(inputMin - inputMax) < FLT_EPSILON){
		ofLogWarning("ofMath") << "ofMap(): avoiding possible divide by zero, \
			check inputMin and inputMax: " << inputMin << " " << inputMax;
		return tensor(-1);
	} else {

		// outVal = ((value - inputMin) / (inputMax - inputMin)
		// outVal = outVal * (outputMax - outputMin) + outputMin);

		float divider = inputMax - inputMin;
		float multiplier = outputMax - outputMin;

		auto result = sub(inputTensor, inputMin);
		result = div(result, divider);
		result = mul(result, multiplier);
		result = add(result, outputMin);
	
		return result;
	}


}

};
