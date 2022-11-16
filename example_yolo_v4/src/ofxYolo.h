/*
 * Example made with love by Jonathan Frank 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 *
 * Originally from ofxTensorFlow2 example_yolo_v4 under a
 * BSD Simplified License: https://github.com/zkmkarlsruhe/ofxTensorFlow2
 */

#pragma once

#include "ofxTensorFlow2.h"
#include "ofFileUtils.h"

/// \class ofxYOLO
/// \brief wrapper for the YOLOv4 realtime object recognition model
///
/// this model works with the COCO dataset via a textfile with one class string
/// per line
///
/// the model accepts a single input image only, the image is automatically
/// resized to the expected input size internally
///
/// basic usage example:
///
/// class ofApp : public ofBaseApp {
/// public:
/// ...
///     ofxYolo yolo;
/// };
///
/// void ofApp::setup() {
///     ...
///     yolo.setup("path/to/modeldir", "path/to/classes.txt");
/// }
///
/// void ofApp.cpp::update() {
///     camera.update();
///     if(camera.isFrameNew()) {
///         yolo.setInput(camera.getPixels());
///     }
///     if(yolo.update()) {
///         ofLogNotice() << "detected objects updated";
///         for(auto object : yolo.getObjects()) {
///            // do something with object
///            ofLog() << "found a " << object.ident.text;
///         }
///     }
/// }
///
class ofxYolo {
	public:

		// model constants
		static const int NN_W = 416; ///< width expected by the model
		static const int NN_H = 416; ///< height expected by the model
		static const int NUM_OBJECTS = 84; ///< number of objects in output

		/// single detected object
		struct Object {
			/// identified object class
			struct Ident {
				int index;          ///< identified class index within classes vector, 0 to size-1
				std::string & text; ///< identified class string, ie. "person", "car", etc
				Ident(int index, std::string & text) : index(index), text(text) {}
			} ident;
			ofRectangle bbox; ///< bounding box, coords within input image size (default) or normalized 0-1
			float confidence; ///< confidence 0-1

			/// create with identified class index and string
			Object(int index, std::string & text) : ident(index, text) {}

			/// draw object bounding box and class info
			void draw() {
				drawBox();
				drawClass();
			}

			/// draw object bounding box
			void drawBox() {
				ofNoFill();
				ofSetColor(ofColor::hotPink);
				ofDrawRectangle(bbox);
			}

			/// draw object class and confidence
			void drawClass() {
				ofSetColor(ofColor::cyan);
				ofDrawBitmapString(ident.text + "\n" + ofToString(confidence, 2),
				                   bbox.x, bbox.y);
			}
		};

		/// ofxTF2::ThreadedModel implementation with custom pre-processing
		class Model : public ofxTF2::ThreadedModel {
			public:
				cppflow::tensor runModel(const cppflow::tensor & input) const override {
					// convert to float image and resize
					auto inputCast = cppflow::cast(input, TF_UINT8, TF_FLOAT);
					inputCast = cppflow::expand_dims(inputCast, 0);
					inputCast = cppflow::mul(inputCast, cppflow::tensor({1/255.f}));
					inputCast = cppflow::resize_bicubic(inputCast, cppflow::tensor({NN_W, NN_H}), true);
					return ofxTF2::Model::runModel(inputCast);
				}
		};

		/// load and set up yolo model & load object class names from a txt file (one name string per line),
		/// returns true on success
		bool setup(const std::string & modelPath="model", const std::string & classPath="classes.txt") {
			
			// model
			if(!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
				ofLogError("ofxYolo") << "failed to set GPU Memory options!";
				return false;
			}
			if(!model.load(modelPath)) {
				return false;
			}
			model.setup({"serving_default_input_1"}, {"StatefulPartitionedCall"});

			// object classes
			ofBuffer buffer = ofBufferFromFile(classPath);
			if(buffer.size() == 0) {
				ofLogError("ofxYolo") << "failed to load " << classPath;
				return false;
			}
			for(auto& line : buffer.getLines()) {
				classes_.push_back(line);
			}

			return true;
		}

		/// clear yolo model and classes
		void clear() {
			model.clear();
			classes_.clear();
		}

		/// set input pixels to process
		void setInput(ofPixels & pixels) {
			input_ = ofxTF2::pixelsToTensor(pixels);
			inputSize_.width = pixels.getWidth();
			inputSize_.height = pixels.getHeight();
			newInput_ = true;
		}

		/// run model on current input, either synchronously by blocking until
		/// finished or asynchronously if background thread is running
		/// returns true if objects are new
		/// note: using the background thread will not block the main thread but
		///       may lead to delayed tracking if the system cannot run the
		///       model quickly enough, in which case cache the input image and
		///       do not set a new input image until the current one is finished
		///       processing
		bool update() {
			if(model.isThreadRunning()) {
				// non-blocking
				if(newInput_ && model.readyForInput()) {
					model.update(input_);
					input_ = cppflow::tensor(0); // clear
					newInput_ = false;
				}
				if(model.isOutputNew()) {
					auto output = model.getOutput();
					parseObjects(output);
					return true;
				}
			}
			else {
				// blocking
				if(newInput_) {
					auto output = model.runModel(input_);
					parseObjects(output);
					newInput_ = false;
					input_ = cppflow::tensor(0); // clear
					return true;
				}
			}
			return false;
		}

		/// draw detected objects within input width & height coordinate system
		void draw() {
			for(auto object : objects) {
				object.draw();
			}
		}

		/// start background thread processing
		void startThread() {
			model.startThread();
		}

		/// stop background thread processing
		void stopThread() {
			model.stopThread();
		}

		/// returns true if background thread is running
		bool isThreadRunning() {return model.isThreadRunning();}

		/// returns a reference to the detected objects, check the confidence
		/// value to determine which are valid, ex. confidence > 0.5, etc
		std::vector<Object> & getObjects() {return objects;}

		/// returns a reference to the detected object class names
		std::vector<std::string> & getClasses() {return classes_;}

		/// returns input width
		/// skeleton positions and bounding box are within this range
		int getWidth() {return inputSize_.width;}

		/// returns input height
		/// skeleton positions and bounding box are within this range
		int getHeight() {return inputSize_.height;}

		/// set minimum object confidence threshold 0-1 (default 0.2),
		/// anything less will be ignored
		void setThreshold(float confidence) {
			threshold_ = ofClamp(confidence, 0, 1);
		}

		/// get the minimum object confidence threshold 0-1
		float getThreshold() {return threshold_;}

		/// set whether to normalize object bounding box coordinates
	    /// * true: within 0-1
	    /// * false: within image size 0-w, 0-h (default)
		void setNormalize(bool normalize) {normalize_ = normalize;}

		/// returns true if object bounding box coordinates are normalized 0-1
	    /// or false if within the image input size 0-w, 0-h
		bool getNormalize() {return normalize_;}

	protected:
		Model model;
		std::vector<Object> objects;

		/// parse tensor output into object data
		void parseObjects(const cppflow::tensor & output) {

			// flatten output tensor to vector
			std::vector<float> vectorOut;
			ofxTF2::tensorToVector(output, vectorOut);

			// parse vector to detected rectangles
			std::vector<float>::const_iterator first;
			std::vector<float>::const_iterator last;
			std::vector<int> maxElementIndexVector;
			std::vector<float> maxElementVector;
			std::vector<std::vector<float>> boundings;
			std::vector<int> rectangleIndices;
			int numRectangles = vectorOut.size() / NUM_OBJECTS;
			std::vector<float> bound;
			for (int i = 0; i < numRectangles; i++) {
				first = vectorOut.begin() + NUM_OBJECTS * i;
				last = vectorOut.begin() + NUM_OBJECTS * i + 4;
				std::vector<float> new_vec(first, last);
				boundings.push_back(new_vec);
				bound.insert(bound.end(), new_vec.begin(), new_vec.end());
				first = vectorOut.begin() + NUM_OBJECTS * i + 4;
				last = vectorOut.begin() + NUM_OBJECTS * i + NUM_OBJECTS;
				std::vector<float> new_vec_id(first, last);
				int max_element_index = std::max_element(new_vec_id.begin(), new_vec_id.end()) - new_vec_id.begin();
				float max_element = new_vec_id[max_element_index];
				maxElementIndexVector.push_back(max_element_index);
				maxElementVector.push_back(max_element);
			}
			cppflow::tensor rectangleTensor = ofxTF2::vectorToTensor(bound, ofxTF2::shapeVector{numRectangles, 4});
			cppflow::tensor maxElementTensor = ofxTF2::vectorToTensor(maxElementVector);
			cppflow::tensor rectangleIndicesTensor = cppflow::non_max_suppression(rectangleTensor, maxElementTensor, 10, 0.5);
			ofxTF2::tensorToVector(rectangleIndicesTensor, rectangleIndices);

			// convert detected rectangles to ofxYolo::Objects
			objects.clear();
			for(int index : rectangleIndices) {
				int classIndex = maxElementIndexVector[index];
				float confidence = maxElementVector[index];
				if(confidence < threshold_) {continue;}
				if(classIndex >= classes_.size()) {
					ofLogWarning("ofxYolo") << "ignoring unknown object class index " << index;
					continue;
				}
				Object object(classIndex, classes_[classIndex]);
				if(normalize_) { // normalized bbox coords
					object.bbox.x = boundings[index][1];
					object.bbox.y = boundings[index][0];
					object.bbox.width = boundings[index][3] - object.bbox.x;
					object.bbox.height = boundings[index][2] - object.bbox.y;
				}
				else { // use input image size
					object.bbox.x = boundings[index][1] * (float)inputSize_.width;
					object.bbox.y = boundings[index][0] * (float)inputSize_.height;
					object.bbox.width = (boundings[index][3] * (float)inputSize_.width) - object.bbox.x;
					object.bbox.height = (boundings[index][2] * (float)inputSize_.height) - object.bbox.y;
				}
				object.confidence = confidence;
				objects.push_back(object);
			}
		}

	private:
		struct Size {
			int width = 1;
			int height = 1;
		} inputSize_; ///< pixel input size
		cppflow::tensor input_; ///< pixel input tensor
		std::vector<std::string> classes_; //< known object classes
		bool newInput_ = false; ///< is the input tensor new?
	    bool normalize_ = false; ///< normalize bounding box coords?
		float threshold_ = 0.2; ///< min object confidence threshold
};
