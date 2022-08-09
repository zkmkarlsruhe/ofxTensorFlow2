/*
 * Example made with love by Jonathhhan 2022
 * https://github.com/Jonathhhan
 * Updated by members of the ZKM | Hertz-Lab 2022
 */

#pragma once

#include "ofxTensorFlow2.h"
#include "ofFileUtils.h"
#include "nms.hpp"

/// \class ofxYOLO
/// \brief wrapper for realtime object recognition on the COCO dataset using the
///        YOLOv4 model
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
			int index; ///< recognized class index within the classes vector
			std::string & ident; ///< recognized class identification
			ofRectangle bbox; ///< bounding box, coords within input image size (default) or normalized 0-1
			float confidence; ///< confidence 0-1

			Object(int index, std::string &ident) : index(index), ident(ident) {}

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
				ofDrawBitmapString(ident + "\n" + ofToString(confidence, 2),
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
		/// value to determine which are valid, ex. confidence > 0.2, etc
		std::vector<Object> & getObjects() {return objects;}

		/// returns a references to the detale object class names
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

			// parse vector to objects
			std::vector<std::vector<float>> boundings;
			std::vector<std::pair<int, float>> id;
			std::vector<float>::const_iterator first;
			std::vector<float>::const_iterator last;
			for(int i = 0; i < vectorOut.size() / NUM_OBJECTS; i++) {
				first = vectorOut.begin() + NUM_OBJECTS * i;
				last = vectorOut.begin() + NUM_OBJECTS * i + 4;
				std::vector<float> newVec(first, last);
				boundings.push_back(newVec);
				first = vectorOut.begin() + NUM_OBJECTS * i + 4;
				last = vectorOut.begin() + NUM_OBJECTS * i + NUM_OBJECTS;
				std::vector<float> newVecId(first, last);
				int maxElementIndex = max_element(newVecId.begin(), newVecId.end()) - newVecId.begin();
				float maxElement = *max_element(newVecId.begin(), newVecId.end());
				id.push_back(std::make_pair(maxElementIndex, maxElement));
			}
			std::vector<std::pair<std::vector<float>, int>> rectangles = nms(boundings, 0.9); // perform non-max regression
			objects.clear();
			for(int i = 0; i < rectangles.size(); i++) {
				float confidence = id[rectangles[i].second].second;
				int index = id[rectangles[i].second].first;
				if(confidence < threshold_) {continue;}
				if(index >= classes_.size()) {
					ofLogWarning("ofxYolo") << "ignoring unknown object class index " << index;
					continue;
				}
				Object object(index, classes_[index]);
				if(normalize_) { // normalized bbox coords
					object.bbox.x = rectangles[i].first[1];
					object.bbox.y = rectangles[i].first[0];
					object.bbox.width = rectangles[i].first[3] - object.bbox.x;
					object.bbox.height = rectangles[i].first[2] - object.bbox.y;
				}
				else { // use input image size
					object.bbox.x = rectangles[i].first[1] * (float)inputSize_.width;
					object.bbox.y = rectangles[i].first[0] * (float)inputSize_.height;
					object.bbox.width = (rectangles[i].first[3] * (float)inputSize_.width) - object.bbox.x;
					object.bbox.height = (rectangles[i].first[2] * (float)inputSize_.height) - object.bbox.y;
				}
				object.confidence = id[rectangles[i].second].second;
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
