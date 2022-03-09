/*
 * ofxTensorFlow2
 *
 * Copyright (c) 2022 ZKM | Hertz-Lab
 * Paul Bethge <bethge@zkm.de>
 * Dan Wilcox <dan.wilcox@zkm.de>
 *
 * BSD Simplified License.
 * For information on usage and redistribution, and for a DISCLAIMER OF ALL
 * WARRANTIES, see the file, "LICENSE.txt," in this distribution.
 *
 * This code has been developed at ZKM | Hertz-Lab as part of „The Intelligent 
 * Museum“ generously funded by the German Federal Cultural Foundation.
 * 
 * This code is based on Memo Akten's ofxMSATensorFlow example-char-rnn:
 * https://github.com/memo/ofxMSATensorFlow
 */

/* Memo Akten writes:

Generative character based Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) demo,
ala Karpathy's char-rnn(http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
and Graves2013(https://arxiv.org/abs/1308.0850).

Models are trained and saved in python with this code (https://github.com/memo/char-rnn-tensorflow)
and loaded in openframeworks for prediction.

I'm supplying a bunch of pretrained models (bible, cooking, erotic, linux, love songs, shakespeare, trump),
and while the text is being generated character by character (at 60fps!) you can switch models in realtime mid-sentence or mid-word.
(Drop more trained models into the folder and they'll be loaded too).

Typing on the keyboard also primes the system, so it'll try and complete based on what you type.

This is a simplified version of what I explain here (https://vimeo.com/203485851), where models can be mixed as well.

Note, all models are trained really quickly with no hyperparameter search or cross validation,
using default architecture of 2 layer LSTM of size 128 with no dropout or any other regularisation.
So they're not great. A bit of hyperparameter tuning would give much better results - but note that would be done in python.
The openframeworks code won't change at all, it'll just load the better model.
*/

#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include <random>

//--------------------------------------------------------------
class ofApp : public ofBaseApp {
public:

	//--------------------------------------------------------------
	void setup();
	void draw();
	void keyPressed(int key);

	void loadModelIndex(int index);
	void loadModel(std::string dir);
	void loadChars(std::string path);
	void runModel(char ch);
	void addChar(char ch);

	// prime model with a sequence of characters
	// this runs the data through the model element by element, so as to update its internal state (stored in t_state)
	// next time we feed the model an element to make a prediction, it will make the prediction primed on this state (i.e. sequence of elements)
	void primeModel(std::string primeData, int primeLength);

	void update();
	void keyReleased(int key);
	void mouseMoved(int x, int y);
	void mouseDragged( int x, int y, int button);
	void mousePressed( int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);

	//--------------------------------------------------------------

	// base model object
	ofxTF2::Model model;

	// for managing character <-> index mapping
	std::vector<char> intToChar;
	std::map<int, char> charToInt;

	// tensors in and out of model
	cppflow::tensor t_dataIn;           // data in
	cppflow::tensor t_state;            // current lstm state
	std::vector<float> lastModelOutput; // probabilities

	// generated text
	std::string textFull;
	std::list<std::string> textLines = { "The" };
	ofTrueTypeFont font;
	unsigned int maxLineWidth = 80;
	unsigned int maxLineNum = 50;
	unsigned int primeLength = 50;
	float sampleTemp = 0.5f;

	// model file management
	ofDirectory modelsDir;          // data/models folder which contains subfolders for each model
	unsigned int curModelIndex = 0; // which model (i.e. folder) we're currently using

	// random generator for sampling
	std::default_random_engine rng;

	// other vars
	bool outputReady = false;
	bool doAutoRun = true;  // auto run every frame
	bool doRunOnce = false; // only run one character
};
