/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#pragma once

#include "ofxTensorFlow2.h"

#define NUM_SKELETONS 6
#define DATA_PER_SKELETON 56
#define BONES_PER_SKELETON 17

#define NOSE 0
#define LEFT_EYE 1
#define RIGHT_EYE 2
#define LEFT_EAR 3
#define RIGHT_EAR 4
#define LEFT_SHOULDER 5
#define RIGHT_SHOULDER 6
#define LEFT_ELBOW 7
#define RIGHT_ELBOW 8
#define LEFT_WRIST 9
#define RIGHT_WRIST 10
#define LEFT_HIP 11
#define RIGHT_HIP 12
#define LEFT_KNEE 13
#define RIGHT_KNEE 14
#define LEFT_ANKLE 15
#define RIGHT_ANKLE 16

struct bone {
	glm::vec3 coords;
	float confidence;
};

class skeleton {
	public:
		bone bones[17];
		ofRectangle bbox;
		float bbox_conf;
};

class ofxMovenet {
	public:

		//--------------------------------------------------------------
		void setup(std::string model_dir) {

			// load the model
			model.load("model");

			// inputs and outputs for the model
			std::vector<string> inputs{"serving_default_input:0"};
			std::vector<string> outputs{"StatefulPartitionedCall:0"};
			model.setup(inputs,outputs);
		}

		//--------------------------------------------------------------
		void update(ofPixels & pxs) {

			// prepare input tensor
			auto input = ofxTF2::pixelsToTensor(pxs);
			input = cppflow::cast(input, TF_UINT8, TF_INT32);
			input = cppflow::expand_dims(input, 0);

			// vector to flattern the tensor output
			std::vector<float> model_results_vector;

			// inference
			auto output = model.runModel(input);

			// flatter output tensor to vector
			ofxTF2::tensorToVector(output, model_results_vector);

			// clear skeletons
			skeletons.clear();

			// parse vector
			for(int i = 0; i < NUM_SKELETONS; i++) {

				int data_index = i * DATA_PER_SKELETON;

				// bounding box and confidence
				float ymin = model_results_vector[data_index+51] * pxs.getHeight();
				float xmin = model_results_vector[data_index+52] * pxs.getWidth();
				float ymax = model_results_vector[data_index+53] * pxs.getHeight();
				float xmax = model_results_vector[data_index+54] * pxs.getWidth();
				float bbconf = model_results_vector[data_index+55];

				skeleton body;

				body.bbox.x = xmin;
				body.bbox.y = ymin;
				body.bbox.width = xmax-xmin;
				body.bbox.height = ymax-ymin;
				body.bbox_conf = bbconf;

				// bones
				for(int j = 0; j < BONES_PER_SKELETON; j++) {
					int bone_index = data_index + (j*3);

					float py = model_results_vector[bone_index] * pxs.getHeight();
					float px = model_results_vector[bone_index + 1] * pxs.getWidth();
					float conf = model_results_vector[bone_index + 2];

					body.bones[j].coords.x = px;
					body.bones[j].coords.y = py;
					body.bones[j].confidence = conf;
				}
				skeletons.push_back(body);
			}
		}

		//--------------------------------------------------------------
		void draw() {
			for(int i = 0; i < skeletons.size(); i++) {
				if(skeletons[i].bbox_conf > 0.5) {

					// pretty bones
					drawPairBones(i, RIGHT_EAR, RIGHT_EYE);
					drawPairBones(i, RIGHT_EYE, NOSE);
					drawPairBones(i, LEFT_EAR, LEFT_EYE);
					drawPairBones(i, LEFT_EYE, NOSE);
					drawPairBones(i, NOSE, RIGHT_SHOULDER);
					drawPairBones(i, NOSE, LEFT_SHOULDER);
					drawPairBones(i, RIGHT_SHOULDER, LEFT_SHOULDER);
					drawPairBones(i, RIGHT_SHOULDER, RIGHT_ELBOW);
					drawPairBones(i, LEFT_SHOULDER, LEFT_ELBOW);
					drawPairBones(i, RIGHT_ELBOW, RIGHT_WRIST);
					drawPairBones(i, LEFT_ELBOW, LEFT_WRIST);
					drawPairBones(i, RIGHT_SHOULDER, RIGHT_HIP);
					drawPairBones(i, LEFT_SHOULDER, LEFT_HIP);
					drawPairBones(i, RIGHT_HIP, LEFT_HIP);
					drawPairBones(i, RIGHT_HIP, RIGHT_KNEE);
					drawPairBones(i, LEFT_HIP, LEFT_KNEE);
					drawPairBones(i, RIGHT_KNEE, RIGHT_ANKLE);
					drawPairBones(i, LEFT_KNEE, LEFT_ANKLE);
				}
			}
		}

		//--------------------------------------------------------------
		std::vector<skeleton> & getSkeletons() {
			return skeletons;
		}

	private:
		ofxTF2::Model model;
		std::vector<skeleton> skeletons;

		//--------------------------------------------------------------
		void drawPairBones(int skeleton_index,int firstBone, int secondBone) {
			ofMesh m;
			m.setMode(OF_PRIMITIVE_LINES);
			m.addVertex(skeletons[skeleton_index].bones[firstBone].coords);
			m.addColor(ofColor::hotPink);
			m.addVertex(skeletons[skeleton_index].bones[secondBone].coords);
			m.addColor(ofColor::purple);
			m.draw();
		}
};
