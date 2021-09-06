/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#pragma once

#include "ofxTensorFlow2.h"

class ofxMovenet {
	public:

		static const int NUM_SKELETONS = 6; //< max number of skeletons tracked
		static const int DATA_PER_SKELETON = 56; //< output data values per skeleton
		static const int BONES_PER_SKELETON = 17; //< bone points per skeleton

		/// single skeleton bone point
		struct Bone {
			glm::vec3 point;  //< 3d position
			float confidence; //< confidence 0-1
		};

		/// single detected skeleton
		struct Skeleton {
			Bone bones[BONES_PER_SKELETON]; //< detected skeleton bone points
			ofRectangle bbox; //< detected skeleton bounding box
			float confidence; //< skeleton bounding box confidence 0-1
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

		//--------------------------------------------------------------
		bool setup(const std::string & modelPath="model") {

			// load the model
			if(!model.load(modelPath)) {
				return false;
			}

			// inputs and outputs for the model
			std::vector<std::string> inputs{"serving_default_input:0"};
			std::vector<std::string> outputs{"StatefulPartitionedCall:0"};
			model.setup(inputs, outputs);

			return true;
		}

		//--------------------------------------------------------------
		void update(ofPixels & pixels) {

			// prepare input tensor
			auto input = ofxTF2::pixelsToTensor(pixels);
			input = cppflow::cast(input, TF_UINT8, TF_INT32);
			input = cppflow::expand_dims(input, 0);

			// inference
			auto output = model.runModel(input);

			// flatten output tensor to vector
			std::vector<float> vectorOut;
			ofxTF2::tensorToVector(output, vectorOut);

			// parse vector to skeletons
			skeletons.clear();
			for(int i = 0; i < NUM_SKELETONS; i++) {
				int d = i * DATA_PER_SKELETON;
				Skeleton skeleton;

				// bounding box and confidence
				float ymin = vectorOut[d+51] * pixels.getHeight();
				float xmin = vectorOut[d+52] * pixels.getWidth();
				float ymax = vectorOut[d+53] * pixels.getHeight();
				float xmax = vectorOut[d+54] * pixels.getWidth();
				skeleton.bbox.x = xmin;
				skeleton.bbox.y = ymin;
				skeleton.bbox.width = xmax - xmin;
				skeleton.bbox.height = ymax - ymin;
				skeleton.confidence = vectorOut[d+55];

				// skeleton
				for(int j = 0; j < BONES_PER_SKELETON; j++) {
					int b = d + (j*3);

					// bone
					float py = vectorOut[b] * pixels.getHeight();
					float px = vectorOut[b + 1] * pixels.getWidth();
					float conf = vectorOut[b + 2];
					skeleton.bones[j].point.x = px;
					skeleton.bones[j].point.y = py;
					skeleton.bones[j].confidence = conf;
				}
				skeletons.push_back(skeleton);
			}
		}

		//--------------------------------------------------------------
		void draw() {
			ofSetLineWidth(2);
			for(auto skeleton : skeletons) {
				if(skeleton.confidence > 0.5) {
					drawPairBones(skeleton, RIGHT_EAR,      RIGHT_EYE);
					drawPairBones(skeleton, RIGHT_EYE,      NOSE);
					drawPairBones(skeleton, LEFT_EAR,       LEFT_EYE);
					drawPairBones(skeleton, LEFT_EYE,       NOSE);
					drawPairBones(skeleton, NOSE,           RIGHT_SHOULDER);
					drawPairBones(skeleton, NOSE,           LEFT_SHOULDER);
					drawPairBones(skeleton, RIGHT_SHOULDER, LEFT_SHOULDER);
					drawPairBones(skeleton, RIGHT_SHOULDER, RIGHT_ELBOW);
					drawPairBones(skeleton, LEFT_SHOULDER,  LEFT_ELBOW);
					drawPairBones(skeleton, RIGHT_ELBOW,    RIGHT_WRIST);
					drawPairBones(skeleton, LEFT_ELBOW,     LEFT_WRIST);
					drawPairBones(skeleton, RIGHT_SHOULDER, RIGHT_HIP);
					drawPairBones(skeleton, LEFT_SHOULDER,  LEFT_HIP);
					drawPairBones(skeleton, RIGHT_HIP,      LEFT_HIP);
					drawPairBones(skeleton, RIGHT_HIP,      RIGHT_KNEE);
					drawPairBones(skeleton, LEFT_HIP,       LEFT_KNEE);
					drawPairBones(skeleton, RIGHT_KNEE,     RIGHT_ANKLE);
					drawPairBones(skeleton, LEFT_KNEE,      LEFT_ANKLE);
				}
			}
			ofSetLineWidth(1);
		}

		//--------------------------------------------------------------
		std::vector<Skeleton> & getSkeletons() {
			return skeletons;
		}

	private:
		ofxTF2::Model model;
		std::vector<Skeleton> skeletons;

		//--------------------------------------------------------------
		void drawPairBones(Skeleton & skeleton, BoneIndex b1, BoneIndex b2) {
			ofMesh m;
			m.setMode(OF_PRIMITIVE_LINES);
			m.addVertex(skeleton.bones[b1].point);
			m.addColor(ofColor::hotPink);
			m.addVertex(skeleton.bones[b2].point);
			m.addColor(ofColor::purple);
			m.draw();
		}
};
