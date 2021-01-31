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
	void downsample(AudioBufferFifo & input, 
					const std::size_t downsamplingFactor){
		
		// get the size of an element
		const int bufferSize = input.front().size();
		const int bufferSizeDownsampled = bufferSize / downsamplingFactor;

		// allocate memory if neccessary
		sample_.resize(input.size() * bufferSizeDownsampled);

		// pop elements from the input buffer, downsample and save to flat buffer
		int i = 0;
		while(!input.empty()){

			// get a buffer from fifo
			const SimpleAudioBuffer & buffer = input.front();

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
			input.pop();
			i++;
		}
	}

	SimpleAudioBuffer sample_;
};
