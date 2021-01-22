#pragma once
    

int testOfxTF2Model(const cppflow::tensor & input, const cppflow::tensor & output){

	// ====== ofxTF2Model ====== //
    ofLog() << "============= Start testing ofxTF2Model =============";

	// load the ofxTF2Model
	ofxTF2Model simpleModel(ofToDataPath("model"));
	
	// reload
	simpleModel.load(ofToDataPath("model"));

	// run
	auto simpleModelOutput = simpleModel.run(input);

	// vector from ofxTensor
	auto vecFromOfxTensor = simpleModelOutput.getVector<float>();

    // compare to output

    return 0;
}
