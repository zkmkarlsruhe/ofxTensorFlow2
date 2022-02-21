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

#include "ofApp.h"


//--------------------------------------------------------------
// from msa::tf:: utilities
template<typename T> vector<T> adjustProbsWithTemp(const vector<T>& p_in, float t) {
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
// from msa::tf:: utilities
template<typename T> int sample_from_prob(std::default_random_engine& rng, const vector<T>& p) {
    std::discrete_distribution<int> rdist (p.begin(),p.end());
    int r = rdist(rng);
    return r;
}


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
	modelsDir.listDir("models");
	if(modelsDir.size()==0) {
		ofLogError() << "Couldn't find models folder.";
		assert(false);
		ofExit(1);
	}
	modelsDir.sort();
	loadModelIndex(0); // load first model

	// seed rng
	rng.seed(ofGetSystemTimeMicros());
	
	// load a font for displaying strings
	font.load(OF_TTF_SANS, 14);
}


//--------------------------------------------------------------
void ofApp::loadModelIndex(int index) {
	curModelIndex = ofClamp(index, 0, modelsDir.size()-1);
	loadModel(modelsDir.getPath(curModelIndex));
}


//--------------------------------------------------------------
void ofApp::loadModel(std::string dir) {

	// load the model, bail out on error
	if(!model.load(dir + "/graph_frz.pb")) {
		std::exit(EXIT_FAILURE);
	}

	// load character map
	loadChars(dir + "/chars.txt");

	// init tensor for input
	// needs to be a single int (index of character)
	// HOWEVER input is not a scalar or vector, but a rank 2 tensor with shape {1, 1} (i.e. a matrix)
	// WHY? because that's how the model was designed to make the internal calculations easier (batch size etc)
	// TBH the model could be redesigned to accept just a rank 1 scalar, and then internally reshaped, but I'm lazy
	t_dataIn = cppflow::fill({1, 1}, 1, TF_INT32);

	// prime model
	primeModel(textFull, primeLength);
}


//--------------------------------------------------------------
void ofApp::loadChars(string path) {
	ofLogVerbose() << "load_chars : " << path;
	intToChar.clear();
	charToInt.clear();
	ofBuffer buffer = ofBufferFromFile(path);

	for(auto line : buffer.getLines()) {
		char c = ofToInt(line); // TODO: will this manage unicode?
		intToChar.push_back(c);
		int i = intToChar.size()-1;
		charToInt[c] = i;
		ofLogVerbose() << i << " : " << c;
	}
}


//--------------------------------------------------------------
void ofApp::primeModel(string primeData, int primeLength) {
	outputReady = false;
	for(unsigned int i=MAX(0, primeData.size()-primeLength); i<primeData.size(); i++) {
		runModel(primeData[i]);
	}
}


//--------------------------------------------------------------
void ofApp::runModel(char ch) {

	t_dataIn = cppflow::fill({1, 1}, (int)charToInt[ch], TF_INT32);

	std::vector<cppflow::tensor> vectorOfInputTensors = {t_dataIn};
	
	if(outputReady) {
		// use state_in if passed in as parameter
		vectorOfInputTensors.push_back(t_state);
	}

	auto vectorOfOutputTensors = model.runMultiModel(vectorOfInputTensors);

	// convert model output from tensors to more manageable types
	if(vectorOfOutputTensors.size() > 1) {
		ofxTF2::tensorToVector<float>(vectorOfOutputTensors[0], lastModelOutput);
		lastModelOutput = adjustProbsWithTemp(lastModelOutput, sampleTemp);

		// save lstm state for next run
		t_state = vectorOfOutputTensors[1];
	}
	outputReady = true;
}


//--------------------------------------------------------------
void ofApp::addChar(char ch) {
	// add sampled char to text
	if(ch == '\n') {
		textLines.push_back("");
	} else {
		textLines.back() += ch;
	}

	// ghetto word wrap
	if(textLines.back().size() > maxLineWidth) {
		std::string textLineCur = textLines.back();
		textLines.pop_back();
		auto last_word_pos = textLineCur.find_last_of(" ");
		textLines.push_back(textLineCur.substr(0, last_word_pos));
		textLines.push_back(textLineCur.substr(last_word_pos));
	}

	// ghetto scroll
	while(textLines.size() > maxLineNum) textLines.pop_front();

	// rebuild text
	textFull.clear();
	for(auto&& text_line : textLines) {
		textFull += "\n" + text_line;
	}

	// feed sampled char back into model
	runModel(ch);
}


//--------------------------------------------------------------
void ofApp::draw() {
	stringstream str;
	str << ofGetFrameRate() << endl;
	str << endl;
	str << "ENTER : toggle auto run " << (doAutoRun ? "(X)" : "( )") << endl;
	str << "RIGHT : sample one char " << endl;
	str << "DEL   : clear text " << endl;
	str << endl;

	str << "Press number key to load model: " << endl;
	for(unsigned int i=0; i<modelsDir.size(); i++) {
		auto marker = (i==curModelIndex) ? ">" : " ";
		str << " " << (i+1) << " : " << marker << " " << modelsDir.getName(i) << endl;
	}

	str << endl;
	str << "Any other key to type," << endl;
	str << "(and prime the model accordingly)" << endl;
	str << endl;


	if(outputReady) {
		// sample one character from probability distribution
		int curCharIndex = sample_from_prob(rng, lastModelOutput);
		char curChar = intToChar[curCharIndex];

		str << "Next char : " << curCharIndex << " | " << curChar << endl;

		if(doAutoRun || doRunOnce) {
			if(doRunOnce) doRunOnce = false;

			addChar(curChar);
		}
	}

	// draw texts
	ofSetColor(150);
	ofDrawBitmapString(str.str(), 20, 20);

	ofSetColor(0, 200, 0);
	ofDrawBitmapString(textFull + "_", 320, 10);
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
		loadModelIndex(key-'1');
		break;

	case OF_KEY_DEL:
		textLines = { "The" };
		break;

	case OF_KEY_RETURN:
		doAutoRun ^= true;
		break;

	case OF_KEY_RIGHT:
		doRunOnce = true;
		doAutoRun = false;
		break;

	default:
		doAutoRun = false;
		if(charToInt.count(key) > 0) addChar(key);
		break;
	}

}


//--------------------------------------------------------------
void ofApp::update() {

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}
