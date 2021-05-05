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
#include <float.h>
#include "ofConstants.h"

#ifdef TARGET_WIN32
/// Windows doesn't provide setenv(), use this _putenv_s() wrapper from:
/// https://stackoverflow.com/a/23616164/2146055
static int setenv(const char *name, const char *value, int overwrite) {
	int errcode = 0;
	if(!overwrite) {
		size_t envsize = 0;
		errcode = getenv_s(&envsize, NULL, 0, name);
		if(errcode || envsize) return errcode;
	}
	return _putenv_s(name, value);
}
#else
#include <sys/errno.h>
#endif

namespace ofxTF2 {

shapeVector getTensorShape(const cppflow::tensor & tensor) {
	return tensor.shape().get_data<shape_t>();
}

bool isSameShape (const shapeVector & lhs, const shapeVector & rhs) {
	if(lhs.size() != rhs.size()) {
			ofLogWarning("ofxTensorFlow2") << "incompatible amount of dimensions "
				<< " for lhs " << std::to_string(lhs.size())
				<< " and rhs " << std::to_string(rhs.size());
			return false;
	}
	for(std::size_t i = 0; i < lhs.size(); i++) {
		if(lhs[i] != rhs[i]) {
			ofLogWarning("ofxTensorFlow2") << "shape mismatch at dimension " << i
				<< " for lhs " << vectorToString(lhs)
				<< " and rhs " << vectorToString(rhs);
			return false;
		}
	}
	return true;
}

cppflow::tensor mapTensorValues(const cppflow::tensor & inputTensor,
	float inputMin, float inputMax, float outputMin, float outputMax) {
	if(fabs(inputMin - inputMax) < FLT_EPSILON){
		ofLogWarning("ofxTensorFlow2") << "avoiding possible divide by zero, "
			<< "check inputMin and inputMax: " << inputMin << " " << inputMax;
		return cppflow::tensor(-1);
	}
	else {
		// outVal = ((value - inputMin) / (inputMax - inputMin)
		// outVal = outVal * (outputMax - outputMin) + outputMin;
		float divider = inputMax - inputMin;
		float multiplier = outputMax - outputMin;
		auto result = cppflow::sub(inputTensor, inputMin);
		result = cppflow::div(result, divider);
		result = cppflow::mul(result, multiplier);
		result = cppflow::add(result, outputMin);
		return result;
	}
}

// the TensorFlow log level is set via the TF_CPP_MIN_LOG_LEVEL env variable,
// TF log const ints are defined in tensorflow/core/platform/default/logging.h
void setLogLevel(ofLogLevel level) {
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
		ofLogError("ofxTensorFlow2") << "unable to set TF_CPP_MIN_LOG_LEVEL: "
			<< errno << " " << strerror(errno);
	}
}

// serialized config options using memory growth
static const std::vector<std::vector<uint8_t>> gpuMemoryGrowPresets {
	{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xb9,0x3f,0x20,0x1},
	{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xc9,0x3f,0x20,0x1},
	{0x32,0xb,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xd3,0x3f,0x20,0x1},
	{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xd9,0x3f,0x20,0x1},
	{0x32,0xb,0x9,0x00,0x00,0x00,0x00,0x00,0x00,0xe0,0x3f,0x20,0x1},
	{0x32,0xb,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xe3,0x3f,0x20,0x1},
	{0x32,0xb,0x9,0x67,0x66,0x66,0x66,0x66,0x66,0xe6,0x3f,0x20,0x1},
	{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xe9,0x3f,0x20,0x1},
	{0x32,0xb,0x9,0xcd,0xcc,0xcc,0xcc,0xcc,0xcc,0xec,0x3f,0x20,0x1},
	{0x32,0xb,0x9,0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0,0x3f,0x20,0x1}
};

// serialized config options without memory growth
static const std::vector<std::vector<uint8_t>> gpuMemoryNoGrowPresets {
	{0x32,0x9,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xb9,0x3f},
	{0x32,0x9,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xc9,0x3f},
	{0x32,0x9,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xd3,0x3f},
	{0x32,0x9,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xd9,0x3f},
	{0x32,0x9,0x9,0x00,0x00,0x00,0x00,0x00,0x00,0xe0,0x3f},
	{0x32,0x9,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xe3,0x3f},
	{0x32,0x9,0x9,0x67,0x66,0x66,0x66,0x66,0x66,0xe6,0x3f},
	{0x32,0x9,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xe9,0x3f},
	{0x32,0x9,0x9,0xcd,0xcc,0xcc,0xcc,0xcc,0xcc,0xec,0x3f},
	{0x32,0x9,0x9,0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0,0x3f}
};

// TODO: check status and return bool?
void setGPUMaxMemory(GPUPercent percent, bool growth) {
	const std::vector<uint8_t> &config =
		(growth ? gpuMemoryGrowPresets[percent] : gpuMemoryNoGrowPresets[percent]);
	setContextOptionsConfig(config);
}

// TODO: check status and return bool?
void setContextOptionsConfig(const std::vector<uint8_t> & config) {
	TFE_ContextOptions *options = TFE_NewContextOptions();
	TFE_ContextOptionsSetConfig(options, config.data(), config.size(),
								cppflow::context::get_status());
	cppflow::get_global_context() = cppflow::context(options);
}

}; // end namespace ofxTF2
