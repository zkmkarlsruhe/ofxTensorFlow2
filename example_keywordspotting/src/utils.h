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

#include "ofxTensorFlow2.h"

#include <queue>
#include <deque>
#include <iostream>


// a simple Fifo with adjustable max length
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
	void setMaxLen(const int32_t maxLength=10) {maxLen = maxLength;}
	private:
	int32_t maxLen;
};

typedef std::vector<float> SimpleAudioBuffer;
typedef FixedFifo<SimpleAudioBuffer> AudioBufferFifo;


class AudioClassifier : public ofxTF2Model {

	public:

	void classify(AudioBufferFifo & bufferFifo, const std::size_t downsamplingFactor,
					int & argMax, float & prob){

		// downsample and empty the incoming Fifo
		downsample(bufferFifo, downsamplingFactor);

		// convert recorded sample to a batch of size one
		ofxTF2::shapeVector tensorShape {1, static_cast<ofxTF2::shape_t>(sample_.size())};
		auto input = ofxTF2::vectorToTensor<float>(sample_, tensorShape);

		// inference
		auto output = runModel(input);

		// convert the output to std::vector
		std::vector<float> outputVector;
		ofxTF2::tensorToVector<float>(output, outputVector);

		// get element with highest probabilty
		auto maxIt = std::max_element(outputVector.begin(), outputVector.end());
		argMax = std::distance(outputVector.begin(), maxIt);
		prob = *maxIt;
	}

	private: 

	// downsample by an integer
	void downsample(AudioBufferFifo & bufferFifo, 
					const std::size_t downsamplingFactor){
		
		// get the size of an element
		const int bufferSize = bufferFifo.front().size();
		const int bufferSizeDownsampled = bufferSize / downsamplingFactor;

		// allocate memory if neccessary
		sample_.resize(bufferFifo.size() * bufferSizeDownsampled);

		// pop elements from the bufferFifo, downsample and save to flat buffer
		int i = 0;
		while(!bufferFifo.empty()){

			// get a buffer from fifo
			const SimpleAudioBuffer & buffer = bufferFifo.front();

			// downsample by integer
			for(int j = 0; j < bufferSizeDownsampled; j++){
				int offset = j * downsamplingFactor;
				float sum = 0.0; 
				for(int k = 0; k < downsamplingFactor; k++){
					sum += buffer[offset+k];
				}
				sample_[i*bufferSizeDownsampled + j] = sum / downsamplingFactor;
			}
			// remove buffer from fifo
			bufferFifo.pop();
			i++;
		}
	}

	SimpleAudioBuffer sample_;
};
