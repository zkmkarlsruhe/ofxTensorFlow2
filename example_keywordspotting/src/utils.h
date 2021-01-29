#pragma once


#include "ofxTensorFlow2.h"

#include <queue>
#include <deque>
#include <iostream>


template <typename T, typename Container=std::deque<T>>
class FixedFifo : public std::queue<T, Container> {
	public:
	FixedFifo(const int32_t maxLength=10) : maxLen(maxLength){}
    void push(const T& value) {
        if (this->size() == maxLen) {
           this->c.pop_front();
        }
        std::queue<T, Container>::push(value);
    }
	private:
	int32_t maxLen;
};

typedef FixedFifo<std::vector<float>> AudioBuffer;


class AudioClassifier : public ofxTF2Model {

	public:

	template<typename T>
	void classify(std::vector<T> sample, int & argMax, float & prob){

		// convert recorded sample to a batch of size one
		ofxTF2::shapeVector tensorShape {1, sample.size()};
		cppflow::tensor input = ofxTF2::vectorToTensor<float>(sample, tensorShape);

		// inference
		cppflow::tensor output = runModel(input);

		// convert the output to std::vector
		std::vector<float> outputVector;
		ofxTF2::tensorToVector<float>(output, outputVector);

		// get element with highest probabilty
		auto maxIt = std::max_element(outputVector.begin(), outputVector.end());
		argMax = std::distance(outputVector.begin(), maxIt);
		prob = *maxIt;
	}

	cppflow::tensor runModel(const cppflow::tensor & input) const {
		// preprocessing: done inside the model (spectrogram, kapre)
		// postprocessing: done inside the model (softmax)		
		return ofxTF2Model::runModel(input); 
	}
};
