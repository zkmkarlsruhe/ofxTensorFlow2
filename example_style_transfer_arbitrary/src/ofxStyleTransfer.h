/*
 * Adapted from example made with love by Jonathan Frank 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 *
 * Originally from ofxTensorFlow2 example_style_transfer_arbitrary under a
 * BSD Simplified License: https://github.com/zkmkarlsruhe/ofxTensorFlow2
 */
#pragma once

#include "ofxTensorFlow2.h"
#include "ofFileUtils.h"

/// \class ofxStyleTransfer
/// \brief wrapper for the arbitrary style transfer model
///
/// the model accepts a style image and applies the style to an input image,
/// the output image will be the same size as the input image
///
/// note: the model requires input input images to be sized in multiples of 32,
///       images are resized between input/output as needed, images must be RGB
///
/// note: input style images are required to 256x256, style images are resized
///       as needed, style images must be RGB
///
/// basic usage example:
///
/// class ofApp : public ofBaseApp {
/// public:
/// ...
///     ofxStyleTransfer styleTransfer;
/// };
///
/// void ofApp::setup() {
///     ...
///     styleTransfer.setup(640, 480, "path/to/modeldir");
///     camera.setup(640, 480);
/// }
///
/// void ofApp.cpp::update() {
///     camera.update();
///     if(camera.isFrameNew()) {
///         styleTransfer.setInput(camera.getPixels());
///     }
///     if(styleTransfer.update()) {
///         ofImage & output = styleTransfer.getOutput();
///         // do something with output
///     }
/// }
///
class ofxStyleTransfer {
	public:

		// model constants
		static const int STYLE_W = 256; ///< style image width expected by the model
		static const int STYLE_H = 256; ///< style image height expected by the model

		/// load and set up style transfer model with input/output image size
		/// returns true on success
		bool setup(int width, int height, const std::string & modelPath="model") {

			// model
			if(!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_90, true)) {
				ofLogError("ofxStyleTransfer") << "failed to set GPU Memory options";
				return false;
			}
			if(!model.load(modelPath)) {
				return false;
			}
			std::vector<std::string> inputNames = {
				"serving_default_placeholder",
				"serving_default_placeholder_1"
			};
			std::vector<std::string> outputNames = {
				"StatefulPartitionedCall"
			};
			model.setup(inputNames, outputNames);

			// input
			inputVector = {cppflow::tensor(0), cppflow::tensor(0)};
			setSize(width, height);

			// output
			outputImage.allocate(size.width, size.height, OF_IMAGE_COLOR);
			return true;
		}

		/// clear model
		void clear() {
			model.clear();
		}

		/// set input pixels to process, resizes as needed
		/// image type must be RGB without alpha
		/// note: set the style image before calling this!
		void setInput(const ofPixels & pixels) {
			cppflow::tensor image = pixelsToFloatTensor(pixels);
			if(pixels.getHeight() != modelSize.width || pixels.getWidth() != modelSize.height) {
				image = cppflow::resize_bicubic(image, cppflow::tensor({size.height, size.width}), true);
			}
			inputVector[0] = image;
			newInput = true;
		}

		/// set input style image, resizes as needed
		/// image type must be RGB without alpha
		void setStyle(const ofPixels & pixels) {
			auto style = pixelsToFloatTensor(pixels);
			if(pixels.getHeight() != STYLE_W || pixels.getWidth() != STYLE_H) {
				style = cppflow::resize_bicubic(style, cppflow::tensor({STYLE_H, STYLE_W}), true);
			}
			inputVector[1] = style;
		}

		/// run model on current input, either synchronously by blocking until
		/// finished or asynchronously if background thread is running
		/// returns true if output image is new
		bool update() {
			if(model.isThreadRunning()) {
				// non-blocking
				if(newInput && model.readyForInput()) {
					model.update(inputVector);
					newInput = false;
					inputVector[0] = cppflow::tensor(0); // clear input image
				}
				if(model.isOutputNew()) {
					auto output = model.getOutputs();
					if(sizeChanged) {
						// reallocate for new input size
						outputImage.allocate(size.width, size.height, OF_IMAGE_COLOR);
						sizeChanged = false;
					}
					if(size.width != outputImage.getWidth() ||
					   size.height != outputImage.getHeight()) {
						// change size in next output frame
						sizeChanged = true;
					}
					if(modelSize.width != outputImage.getWidth() ||
					   modelSize.height != outputImage.getHeight()) {
						resizeTensorToImage(output[0], outputImage);
					}
					floatTensorToImage(output[0], outputImage);
					outputImage.update();
					return true;
				}
			}
			else {
				// blocking
				if(newInput) {
					auto output = model.runMultiModel(inputVector);
					if(modelSize.width != outputImage.getWidth() ||
					   modelSize.height != outputImage.getHeight()) {
						resizeTensorToImage(output[0], outputImage);
					}
					floatTensorToImage(output[0], outputImage);
					outputImage.update();
					newInput = false;
					inputVector[0] = cppflow::tensor(0); // clear input image
					return true;
				}
			}
			return false;
		}

		/// get processed output image
		/// note: output size may differ from getWidth() / getHeight() if
		///       setSize() called while model is processing in non-blocking
		///       background thread
		ofImage & getOutput() {
			return outputImage;
		}

		/// draw current output image
		void draw(float x, float y) {
			outputImage.draw(x, y);
		}

		/// draw current output image
		void draw(float x, float y, float w, float h) {
			outputImage.draw(x, y, w, h);
		}

		/// start background thread processing
		void startThread() {
			sizeChanged = false; // reset change detection
			model.startThread();
		}

		/// stop background thread processing
		void stopThread() {
			model.stopThread();
		}

		/// returns true if background thread is running
		bool isThreadRunning() {return model.isThreadRunning();}

		/// returns input width
		/// note: output width may differ if setSize() called while model is
		///       processing in non-blocking background thread, in which case
		///       check the outputImage size
		int getWidth() {return size.width;}

		/// returns input height
		/// note: output height may differ if setSize() called while model is
		///       processing in non-blocking background thread, in which case
		///       check the outputImage size
		int getHeight() {return size.height;}

		/// set new input size
		void setSize(int width, int height) {
			size.width = width;
			size.height = height;
			modelSize.width = ofxStyleTransfer::roundupto(width, 32);
			modelSize.height = ofxStyleTransfer::roundupto(height, 32);
			//if(modelSize.width != width || modelSize.height != height) {
			//	ofLogWarning("ofxStyleTransfer") << width << "x" << height
			//		<< " not multiple(s) of 32, rounding up to "
			//		<< modelSize.width << "x" << modelSize.height;
			//}
			if(model.isThreadRunning() && model.readyForInput()) {
				// resize output image if not processing in background thread
				sizeChanged = true;
			}
		}

		// round n up to nearest multiple, positive only
		static int roundupto(int n, int multiple) {
			return n + multiple - 1 - (n + multiple - 1) % multiple;
		}

	protected:
		ofxTF2::ThreadedModel model;

		// convert ofPixels to a float image tensor
		cppflow::tensor pixelsToFloatTensor(const ofPixels & pixels) {
			auto t = ofxTF2::pixelsToTensor(pixels);
			t = cppflow::expand_dims(t, 0);
			t = cppflow::cast(t, TF_UINT8, TF_FLOAT);
			t = cppflow::mul(t, cppflow::tensor({1.0f / 255.f}));
			return t;
		}

		// convert float image tensor to ofImage
		void floatTensorToImage(cppflow::tensor tensor, ofImage & image) {
			tensor = cppflow::mul(tensor, cppflow::tensor({255.f}));
			tensor = cppflow::cast(tensor, TF_FLOAT, TF_UINT8);
			ofxTF2::tensorToImage(tensor, image);
		}

		// resize tensor to match ofImage
		void resizeTensorToImage(cppflow::tensor & tensor, ofImage & image) {
			tensor = cppflow::resize_bicubic(tensor,
				cppflow::tensor({(int)outputImage.getHeight(),
							     (int)outputImage.getWidth()}), true);
		}

	private:

		struct Size {
			int width = 1;
			int height = 1;
		};
		struct Size size; ///< pixel input (& output) size
		struct Size modelSize; ///< pixel size for the model, multiples of 32
		std::vector<cppflow::tensor> inputVector; // {input image, style image}
		ofImage outputImage; ///< output image
		bool newInput = false; ///< is the input tensor new?

		/// size change between input & output? used when non-blocking only
		bool sizeChanged = false;
};
