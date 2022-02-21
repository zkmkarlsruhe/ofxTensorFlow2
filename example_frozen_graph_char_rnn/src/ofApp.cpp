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
 * 
 * This code is based on Memo Akten's ofxMSATensorFlow example.
 */

#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetColor(255);
	ofBackground(0);
	ofSetVerticalSync(true);
	ofSetLogLevel(OF_LOG_VERBOSE);
	ofSetWindowTitle("example_frozen_graph_char_rnn");
	ofSetFrameRate(20); // generating a character per frame at 60fps is too fast to read in realtime

	// set model type and i/o names
	model.setModelType(cppflow::model::TYPE::FROZEN_GRAPH);
	std::vector<std::string> inputNames = {
		"data_in",
		"state_in",
	};
	std::vector<std::string> outputNames = {
		"data_out",
		"state_out",
	};
	model.setup(inputNames, outputNames);


	// scan models dir
	models_dir.listDir("models");
	if(models_dir.size()==0) {
		ofLogError() << "Couldn't find models folder.";
		assert(false);
		ofExit(1);
	}
	models_dir.sort();
	load_model_index(0); // load first model

	// seed rng
	rng.seed(ofGetSystemTimeMicros());
	
	// load a font for displaying strings
	font.load(OF_TTF_SANS, 14);
}


//--------------------------------------------------------------
// Load model by folder INDEX
void ofApp::load_model_index(int index) {
	cur_model_index = ofClamp(index, 0, models_dir.size()-1);
	load_model(models_dir.getPath(cur_model_index));
}


//--------------------------------------------------------------
// Load graph (model trained in and exported from python) by folder NAME, and initialize session
void ofApp::load_model(std::string dir) {

	// TODO load model from 'dir'
	// load the model, bail out on error
	const std::string model_path = dir + "/graph_frz.pb";
	if(!model.load(model_path)) {
		std::exit(EXIT_FAILURE);
	}

	// load character map
	// TODO load model from 'dir'
	const std::string chars_path = dir + "/chars.txt";
	load_chars(chars_path);

	// init tensor for input
	// needs to be a single int (index of character)
	// HOWEVER input is not a scalar or vector, but a rank 2 tensor with shape {1, 1} (i.e. a matrix)
	// WHY? because that's how the model was designed to make the internal calculations easier (batch size etc)
	// TBH the model could be redesigned to accept just a rank 1 scalar, and then internally reshaped, but I'm lazy
	t_data_in = cppflow::fill({1, 1}, 1, TF_INT32);

	// prime model
	prime_model(text_full, prime_length);
}


//--------------------------------------------------------------
// load character <-> index mapping
void ofApp::load_chars(string path) {
	ofLogVerbose() << "load_chars : " << path;
	int_to_char.clear();
	char_to_int.clear();
	ofBuffer buffer = ofBufferFromFile(path);

	for(auto line : buffer.getLines()) {
		char c = ofToInt(line); // TODO: will this manage unicode?
		int_to_char.push_back(c);
		int i = int_to_char.size()-1;
		char_to_int[c] = i;
		ofLogVerbose() << i << " : " << c;
	}
}



//--------------------------------------------------------------
// prime model with a sequence of characters
// this runs the data through the model element by element, so as to update its internal state (stored in t_state)
// next time we feed the model an element to make a prediction, it will make the prediction primed on this state (i.e. sequence of elements)
void ofApp::prime_model(string prime_data, int prime_length) {

	ofLogVerbose() << "prime_model : " << prime_data << " (" << prime_length << ")";

	outputReady = false;

	for(int i=MAX(0, prime_data.size()-prime_length); i<prime_data.size(); i++) {
		run_model(prime_data[i]);
	}
}


template<typename T> vector<T> adjust_probs_with_temp(const vector<T>& p_in, float t) {
    if(t>0) {
        vector<T> p_out(p_in.size());
        T sum = 0;
        for(size_t i=0; i<p_in.size(); i++) {
            p_out[i] = exp( log((double)p_in[i]) / (double)t );
            sum += p_out[i];
        }

        if(sum > 0)
            for(size_t i=0; i<p_out.size(); i++) p_out[i] /= sum;

        return p_out;
    }

    return p_in;
}

//--------------------------------------------------------------
// run model on a single character
void ofApp::run_model(char ch) {

	ofLogVerbose() << "run_model : " << ch << " (" << char_to_int[ch] << ")";


	// copy input data into tensor
	// ofxTF2::vectorToTensor<int32_t>(std::vector<int32_t>(char_to_int[ch]));
	// t_data_in = {char_to_int[ch]};

	t_data_in = cppflow::fill({1, 1}, (int)char_to_int[ch], TF_INT32);

	std::vector<cppflow::tensor> vectorOfInputTensors = {t_data_in};
	
	if(outputReady) {
		// use state_in if passed in as parameter
		vectorOfInputTensors.push_back(t_state);
		ofLogVerbose() << "state_in is not empty";
	}
	else {
		ofLogVerbose() << "state_in is empty";
	}

	auto vectorOfOutputTensors = model.runMultiModel(vectorOfInputTensors);

	// convert model output from tensors to more manageable types
	if(vectorOfOutputTensors.size() > 1) {
		ofxTF2::tensorToVector<float>(vectorOfOutputTensors[0], last_model_output);
		last_model_output = adjust_probs_with_temp(last_model_output, sample_temp);

		// save lstm state for next run
		t_state = vectorOfOutputTensors[1];
	}
	outputReady = true;
}


//--------------------------------------------------------------
// add character to string, manage ghetto wrapping for display, run model etc.
void ofApp::add_char(char ch) {
	// add sampled char to text
	if(ch == '\n') {
		text_lines.push_back("");
	} else {
		text_lines.back() += ch;
	}

	// ghetto word wrap
	if(text_lines.back().size() > max_line_width) {
		string text_line_cur = text_lines.back();
		text_lines.pop_back();
		auto last_word_pos = text_line_cur.find_last_of(" ");
		text_lines.push_back(text_line_cur.substr(0, last_word_pos));
		text_lines.push_back(text_line_cur.substr(last_word_pos));
	}

	// ghetto scroll
	while(text_lines.size() > max_line_num) text_lines.pop_front();

	// rebuild text
	text_full.clear();
	for(auto&& text_line : text_lines) {
		text_full += "\n" + text_line;
	}

	// feed sampled char back into model
	run_model(ch);
}


template<typename T> int sample_from_prob(std::default_random_engine& rng, const vector<T>& p) {
    std::discrete_distribution<int> rdist (p.begin(),p.end());
    int r = rdist(rng);
    return r;
}

//--------------------------------------------------------------
void ofApp::draw() {
	stringstream str;
	str << ofGetFrameRate() << endl;
	str << endl;
	str << "ENTER : toggle auto run " << (do_auto_run ? "(X)" : "( )") << endl;
	str << "RIGHT : sample one char " << endl;
	str << "DEL   : clear text " << endl;
	str << endl;

	str << "Press number key to load model: " << endl;
	for(int i=0; i<models_dir.size(); i++) {
		auto marker = (i==cur_model_index) ? ">" : " ";
		str << " " << (i+1) << " : " << marker << " " << models_dir.getName(i) << endl;
	}

	str << endl;
	str << "Any other key to type," << endl;
	str << "(and prime the model accordingly)" << endl;
	str << endl;


	if(outputReady) {
		// sample one character from probability distribution
		int cur_char_index = sample_from_prob(rng, last_model_output);
		char cur_char = int_to_char[cur_char_index];

		str << "Next char : " << cur_char_index << " | " << cur_char << endl;

		if(do_auto_run || do_run_once) {
			if(do_run_once) do_run_once = false;

			add_char(cur_char);
		}
	}

	// // display probability histogram
	// msa::tf::draw_probs(last_model_output, ofRectangle(0, 0, ofGetWidth(), ofGetHeight()));


	// draw texts
	ofSetColor(150);
	ofDrawBitmapString(str.str(), 20, 20);

	ofSetColor(0, 200, 0);
	ofDrawBitmapString(text_full + "_", 320, 10);
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch(key) {
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
	case '7':
	case '8':
	case '9':
		load_model_index(key-'1');
		break;

	case OF_KEY_DEL:
		text_lines = { "The" };
		break;

	case OF_KEY_RETURN:
		do_auto_run ^= true;
		break;

	case OF_KEY_RIGHT:
		do_run_once = true;
		do_auto_run = false;
		break;

	default:
		do_auto_run = false;
		if(char_to_int.count(key) > 0) add_char(key);
		break;
	}

}




// //--------------------------------------------------------------
// void ofApp::keyReleased(int key) {

// }

// //--------------------------------------------------------------
// void ofApp::mouseMoved(int x, int y) {

// }

// //--------------------------------------------------------------
// void ofApp::mouseDragged(int x, int y, int button) {

// }

// //--------------------------------------------------------------
// void ofApp::mousePressed(int x, int y, int button) {

// }

// //--------------------------------------------------------------
// void ofApp::mouseReleased(int x, int y, int button) {

// }

// //--------------------------------------------------------------
// void ofApp::mouseEntered(int x, int y) {

// }

// //--------------------------------------------------------------
// void ofApp::mouseExited(int x, int y) {

// }

// //--------------------------------------------------------------
// void ofApp::windowResized(int w, int h) {

// }

// //--------------------------------------------------------------
// void ofApp::gotMessage(ofMessage msg) {

// }

// //--------------------------------------------------------------
// void ofApp::dragEvent(ofDragInfo dragInfo) {

// }
