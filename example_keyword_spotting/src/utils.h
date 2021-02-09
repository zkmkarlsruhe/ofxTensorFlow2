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

		FixedFifo(const std::size_t maxLength=10) : maxLen(maxLength) {}

		void push(const T& value) {
			if(this->size() == maxLen) {
				this->c.pop_front();
			}
			std::queue<T, Container>::push(value);
		}

		void setMaxLen(const std::size_t maxLength) {
			maxLen = maxLength;
		}

	private:
		std::size_t maxLen;
};

typedef std::vector<float> SimpleAudioBuffer;
typedef FixedFifo<SimpleAudioBuffer> AudioBufferFifo;

// custom ofxTF2::Model implementation to handle audio sample conversion, etc
class AudioClassifier : public ofxTF2::Model {

	public:

		void classify(AudioBufferFifo & bufferFifo, const std::size_t downsamplingFactor,
					  int & argMax, float & prob) {

			SimpleAudioBuffer sample;

			// downsample and empty the incoming Fifo
			downsample(bufferFifo, sample, downsamplingFactor);

			// convert recorded sample to a batch of size one
			ofxTF2::shapeVector tensorShape {1, static_cast<ofxTF2::shape_t>(sample.size())};
			auto input = ofxTF2::vectorToTensor<float>(sample, tensorShape);

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
		void downsample(AudioBufferFifo & bufferFifo, SimpleAudioBuffer & sample,
						const std::size_t downsamplingFactor) {

			// get the size of an element
			const std::size_t bufferSize = bufferFifo.front().size();
			const std::size_t bufferSizeDownsampled = bufferSize / downsamplingFactor;

			// allocate memory if neccessary
			sample.resize(bufferFifo.size() * bufferSizeDownsampled);

			// pop elements from the bufferFifo, downsample and save to flat buffer
			std::size_t i = 0;
			while(!bufferFifo.empty()) {

				// get a buffer from fifo
				const SimpleAudioBuffer & buffer = bufferFifo.front();

				// downsample by integer
				for(std::size_t j = 0; j < bufferSizeDownsampled; j++) {
					std::size_t offset = j * downsamplingFactor;
					float sum = 0.0;
					for(std::size_t k = 0; k < downsamplingFactor; k++) {
						sum += buffer[offset+k];
					}
					sample[i*bufferSizeDownsampled + j] = sum / downsamplingFactor;
				}
				// remove buffer from fifo
				bufferFifo.pop();
				i++;
			}
		}
};
