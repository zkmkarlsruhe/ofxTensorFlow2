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
							{w, h, 1});
		case OF_IMAGE_COLOR:
			return tensor(
			std::vector<float>(pixels.begin(),
								pixels.end()),
							{w, h, 3});
		case OF_IMAGE_COLOR_ALPHA:
			return tensor(
			std::vector<float>(pixels.begin(),
								pixels.end()),
							{w, h, 4});
		case OF_IMAGE_UNDEFINED:
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
			return ofLogError() << "tensor_to_pixels, pixels image type undefined";
	}

}

};
