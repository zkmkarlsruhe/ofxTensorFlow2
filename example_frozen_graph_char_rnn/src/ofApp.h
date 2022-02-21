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
 * This code is based on Memo Akten's ofxMSATensorFlow example.
 */


/* From Memo Akten's ofxMSATensorFlow example */
/*
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

    // Load model by folder INDEX
    void load_model_index(int index);

    // Load graph (model trained in and exported from python) by folder NAME, and initialize session
    void load_model(string dir);

    // load character <-> index mapping
    void load_chars(string path);

    // prime model with a sequence of characters
    // this runs the data through the model element by element, so as to update its internal state (stored in t_state)
    // next time we feed the model an element to make a prediction, it will make the prediction primed on this state (i.e. sequence of elements)
    void prime_model(string prime_data, int prime_length);

    // run model on a single character
    void run_model(char ch);

    void add_char(char ch);

    //--------------------------------------------------------------

    // for managing character <-> index mapping
    vector<char> int_to_char;
    map<int, char> char_to_int;

    // tensors in and out of model
    cppflow::tensor t_data_in;   // data in
    cppflow::tensor t_state;     // current lstm state
    vector<float> last_model_output;    // probabilities

    // generated text
    // managing word wrap in very ghetto way
    string text_full;
    list<string> text_lines = { "The" };
    int max_line_width = 120;
    int max_line_num = 50;

    // model file management
    ofDirectory models_dir;    // data/models folder which contains subfolders for each model
    int cur_model_index = 0; // which model (i.e. folder) we're currently using

    // random generator for sampling
    std::default_random_engine rng;

    // other vars
    int prime_length = 50;
    float sample_temp = 0.5f;

    bool outputReady = false;
    bool do_auto_run = true;    // auto run every frame
    bool do_run_once = false;   // only run one character

	ofxTF2::Model model;
	std::vector<float> inputVector;
	std::vector<float> outputVector;
	ofTrueTypeFont font;

};
