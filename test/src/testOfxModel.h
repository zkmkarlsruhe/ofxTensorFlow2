#pragma once
    

int testOfxModel(const cppflow::tensor & input, const cppflow::tensor & output){

	// ====== ofxModel ====== //
    ofLog() << "============= Start testing ofxModel =============";

	// load the ofxModel
	ofxModel simpleModel(ofToDataPath("model"));
	
	// reload
	simpleModel.load(ofToDataPath("model"));

	// run
	auto simpleModelOutput = simpleModel.run(input);

	// vector from ofxTensor
	auto vecFromOfxTensor = simpleModelOutput.getVector<float>();

    // compare to output

    return 0;
}
