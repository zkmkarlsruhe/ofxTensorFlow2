#pragma once

template<typename T>
void printVector(const std::vector<T> & vec);

int testOfxTensor(){

	// ====== ofxTensor ====== //
    ofLog() << "============= Start testing ofxTensor =============";

	// ofxTensor from cppflow::tensor
	auto cppflowTensor = cppflow::fill({1, 2, 2, 3}, 2);
	ofxTensor fromCppflowTensor (cppflowTensor);

    // check comparision operator
	auto cppflowTensorOtherShape = cppflow::fill({2, 1, 2, 3}, 0.9f);
	auto cppflowTensorOtherType = cppflow::fill({1, 2, 2, 3}, 1);
    ofLog() << "is same self: " << (fromCppflowTensor == cppflowTensor);
    ofLog() << "is same shape: " << (fromCppflowTensor == cppflowTensorOtherShape);
    ofLog() << "is same type: " << (fromCppflowTensor == cppflowTensorOtherType);

    // check value 
	auto cppflowTensorOtherValue = cppflow::fill({1, 2, 2, 3}, 0.99f);
    bool isEqualOtherValue = fromCppflowTensor.equals<float>(cppflowTensorOtherValue);
    bool isEqualSelf = fromCppflowTensor.equals<float>(cppflowTensor);
    ofLog() << "equal to other value: " << isEqualOtherValue;
    ofLog() << "equal to self: " << isEqualSelf;

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

    printVector(vecFromOfxTensor);

    return 0;
}

template<typename T>
void printVector(const std::vector<T> & vec){

    std::string out;
	for (int i=0; i < vec.size(); i++){
		out.append(std::to_string(vec[i]));
        out.append(" ");
	}
	ofLog() << out;
}

// void printTensor(const cppflow::tensor & tensor){

//     std::string out;
// 	for (int i=0; i < tensor.size(); i++){
// 		out.append(std::to_string(vecFromOfxTensor[i]));
//         out.append(" ");
// 	}
// 	ofLog() << out;
// }