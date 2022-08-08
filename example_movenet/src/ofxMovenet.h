/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#pragma once

#include "ofxTensorFlow2.h"

/// \class ofxMovenet
/// \brief wrapper for the movenet multi pose estimation model
///
/// note: input width and height must be multiples of 32, it is recommended that
///       the larger dimension is a multiple of 256
///
/// the model accepts a single input image only
///
/// basic usage example:
///
/// class ofApp : public ofBaseApp {
/// public:
/// ...
///     std::size_t nnWidth = 512;
///     std::size_t nnHeight = 288;
///     ofxMovenet movenet;
/// };
///
/// void ofApp::setup() {
///     ...
///     movenet.setup("path/to/modeldir");
/// }
///
/// void ofApp.cpp::update() {
///     camera.update();
///     if(camera.isFrameNew()) {
///         ofPixels pixels(camera.getPixels());
///         pixels.resize(nnWidth, nnHeight);
///         imgOut.setFromPixels(pixels);
///         imgOut.update();
///         movenet.setInput(pixels);
///     }
///     if(movenet.update()) {
///         ofLogNotice() << "skeletons updated";
///     }
/// }
///
class ofxMovenet {
	public:

		static const int NUM_SKELETONS = 6; ///< max skeletons tracked
		static const int DATA_PER_SKELETON = 56; ///< output values per skeleton
		static const int BONES_PER_SKELETON = 17; ///< bone points per skeleton

		/// single skeleton bone point
		struct Bone {
			glm::vec3 point;  ///< 3d position
			float confidence; ///< confidence 0-1
		};

		/// skeleton bone point index
		enum BoneIndex {
			NOSE           = 0,
			LEFT_EYE       = 1,
			RIGHT_EYE      = 2,
			LEFT_EAR       = 3,
			RIGHT_EAR      = 4,
			LEFT_SHOULDER  = 5,
			RIGHT_SHOULDER = 6,
			LEFT_ELBOW     = 7,
			RIGHT_ELBOW    = 8,
			LEFT_WRIST     = 9,
			RIGHT_WRIST    = 10,
			LEFT_HIP       = 11,
			RIGHT_HIP      = 12,
			LEFT_KNEE      = 13,
			RIGHT_KNEE     = 14,
			LEFT_ANKLE     = 15,
			RIGHT_ANKLE    = 16
		};

		/// single detected skeleton
		struct Skeleton {
			Bone bones[BONES_PER_SKELETON]; ///< detected skeleton bone points
			ofRectangle bbox; ///< detected skeleton bounding box
			float confidence; ///< skeleton bounding box confidence 0-1

			/// draw a line between two bone points
			void drawBone(BoneIndex b1, BoneIndex b2) {
				ofMesh m;
				m.setMode(OF_PRIMITIVE_LINES);
				m.addVertex(bones[b1].point);
				m.addColor(ofColor::hotPink);
				m.addVertex(bones[b2].point);
				m.addColor(ofColor::purple);
				m.draw();
			}

			/// draw skeleton as lines between bone points
			void draw() {
				drawBone(RIGHT_EAR,      RIGHT_EYE);
				drawBone(RIGHT_EYE,      NOSE);
				drawBone(LEFT_EAR,       LEFT_EYE);
				drawBone(LEFT_EYE,       NOSE);
				drawBone(NOSE,           RIGHT_SHOULDER);
				drawBone(NOSE,           LEFT_SHOULDER);
				drawBone(RIGHT_SHOULDER, LEFT_SHOULDER);
				drawBone(RIGHT_SHOULDER, RIGHT_ELBOW);
				drawBone(LEFT_SHOULDER,  LEFT_ELBOW);
				drawBone(RIGHT_ELBOW,    RIGHT_WRIST);
				drawBone(LEFT_ELBOW,     LEFT_WRIST);
				drawBone(RIGHT_SHOULDER, RIGHT_HIP);
				drawBone(LEFT_SHOULDER,  LEFT_HIP);
				drawBone(RIGHT_HIP,      LEFT_HIP);
				drawBone(RIGHT_HIP,      RIGHT_KNEE);
				drawBone(LEFT_HIP,       LEFT_KNEE);
				drawBone(RIGHT_KNEE,     RIGHT_ANKLE);
				drawBone(LEFT_KNEE,      LEFT_ANKLE);
			}
		};

		/// custom ofxTF2::ThreadedModel implementation with custom pre-processing
		class Model : public ofxTF2::ThreadedModel {
			public:
				cppflow::tensor runModel(const cppflow::tensor & input) const override {
					auto inputCast = cppflow::cast(input, TF_UINT8, TF_INT32);
					inputCast = cppflow::expand_dims(inputCast, 0);
					return ofxTF2::Model::runModel(inputCast);
				}
		};

		/// load and set up movenet model, returns true on success
		bool setup(const std::string & modelPath="model") {
			if(!model.load(modelPath)) {
				return false;
			}
			std::vector<std::string> inputs{"serving_default_input:0"};
			std::vector<std::string> outputs{"StatefulPartitionedCall:0"};
			model.setup(inputs, outputs);
			return true;
		}

		/// clear movenet model
		void clear() {
			model.clear();
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
		/// returns true if skeletons are new
		/// note: using the background thread will not block the main thread but
		///       may lead to delayed tracking if the system cannot run the model
		///       quickly enough
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
					parseSkeletons(output);
					return true;
				}
			}
			else {
				// blocking
				if(newInput_) {
					auto output = model.runModel(input_);
					parseSkeletons(output);
					newInput_ = false;
					input_ = cppflow::tensor(0); // clear
					return true;
				}
			}
			return false;
		}

		/// draw detected skeletons within input width & height coordinate system
		void draw() {
			ofSetLineWidth(2);
			for(auto skeleton : skeletons) {
				if(skeleton.confidence > 0.5) {
					skeleton.draw();
				}
			}
			ofSetLineWidth(1);
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

		/// returns a reference to the detected skeletons, check the confidence
		/// value to determine which are valid, ex. confidence > 0.5, etc
		std::vector<Skeleton> & getSkeletons() {return skeletons;}

		/// returns input width
		/// skeleton positions and bounding box are within this range
		int getWidth() {return inputSize_.width;}

		/// returns input height
		/// skeleton positions and bounding box are within this range
		int getHeight() {return inputSize_.height;}

	protected:
		Model model;
		std::vector<Skeleton> skeletons;

		/// parse tensor output into skeleton data
		void parseSkeletons(const cppflow::tensor & output) {

			// flatten output tensor to vector
			std::vector<float> vectorOut;
			ofxTF2::tensorToVector(output, vectorOut);

			// parse vector to skeletons
			skeletons.clear();
			for(int i = 0; i < NUM_SKELETONS; i++) {
				int d = i * DATA_PER_SKELETON;
				Skeleton skeleton;

				// bounding box and confidence
				float ymin = vectorOut[d+51] * inputSize_.height;
				float xmin = vectorOut[d+52] * inputSize_.width;
				float ymax = vectorOut[d+53] * inputSize_.height;
				float xmax = vectorOut[d+54] * inputSize_.width;
				skeleton.bbox.x = xmin;
				skeleton.bbox.y = ymin;
				skeleton.bbox.width = xmax - xmin;
				skeleton.bbox.height = ymax - ymin;
				skeleton.confidence = vectorOut[d+55];

				// skeleton
				for(int j = 0; j < BONES_PER_SKELETON; j++) {
					int b = d + (j * 3);

					// bone
					float py = vectorOut[b] * inputSize_.height;
					float px = vectorOut[b+1] * inputSize_.width;
					float conf = vectorOut[b+2];
					skeleton.bones[j].point.x = px;
					skeleton.bones[j].point.y = py;
					skeleton.bones[j].confidence = conf;
				}
				skeletons.push_back(skeleton);
			}
		}

	private:
		struct Size {
			int width = 1;
			int height = 1;
		} inputSize_; ///< pixel input size
		cppflow::tensor input_; ///< pixel input tensor
		bool newInput_ = false; ///< is the input tensor new?
};
