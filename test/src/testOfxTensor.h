#pragma once

int testOfxTensor(){

	// ====== ofxTensor ====== //

	// ofxTensor from cppflow::tensor
	auto cppflowTensor = cppflow::fill({1, 2, 2, 3}, 2);
	ofxTensor fromCppflowTensor (cppflowTensor);

    // check comparision operator
	auto cppflowTensorOtherShape = cppflow::fill({2, 1, 2, 3}, 0.9f);
	auto cppflowTensorOtherType = cppflow::fill({1, 2, 2, 3}, 1);
    ofLog() << "is same self: " << fromCppflowTensor == cppflowTensor;
    ofLog() << "is same shape: " << fromCppflowTensor == cppflowTensorOtherShape;
    ofLog() << "is same type: " << fromCppflowTensor == cppflowTensorOtherType;

    // check value 
	auto cppflowTensorOtherValue = cppflow::fill({1, 2, 2, 3}, 0.99f);
    bool isEqualOtherValue = fromCppflowTensor.equals<float>(cppflowTensorOtherValue);
    bool isEqualSelf = fromCppflowTensor.equals<float>(cppflowTensor);
    ofLog() << "equal to other value: " << isEqualOtherValue;
    ofLog() << "equal to self: " << isEqualOtherValue;

    // todo ostream operator not working
    // std::cout << fromCppflowTensor << std::endl;
    // ofLog() << fromCppflowTensor.getTensor();

	// todo ofxTensor from vector
	// todo ofxTensor from ofImage
	// todo ofxTensor from ofPixels

	// vector from ofxTensor
	auto vecFromOfxTensor = fromCppflowTensor.getVector<int>();

	// todo vector from ofxTensor
	// todo ofImage from ofxTensor  
	// todo ofPixels from ofxTensor

	// std::cout << "vecFromOfxTensor size: " << vecFromOfxTensor.size() << std::endl;
	// std::cout << "vecFromCppflowTensor size: " << input.get_data<float>().size() << std::endl;

	for (int i=0; i < vecFromOfxTensor.size(); i++){
		std::cout << vecFromOfxTensor[i] << ", ";
	}
	std::cout.flush();

    return 0;
}