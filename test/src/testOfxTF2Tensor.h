#pragma once

template<typename T>
void printVector(const std::vector<T> & vec);

int testOfxTF2Tensor(){

	// ====== ofxTF2Tensor ====== //
    ofLog() << "============= Start testing ofxTF2Tensor =============";

	// ofxTF2Tensor from cppflow::tensor
	auto cppflowTensor = cppflow::fill({1, 2, 2, 3}, 2);
	ofxTF2Tensor fromCppflowTensor (cppflowTensor);

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

	// todo ofxTF2Tensor from vector
	// todo ofxTF2Tensor from ofImage
	// todo ofxTF2Tensor from ofPixels

	// vector from ofxTF2Tensor
	auto vecFromofxTF2Tensor = fromCppflowTensor.getVector<int>();

	// todo vector from ofxTF2Tensor
	// todo ofImage from ofxTF2Tensor  
	// todo ofPixels from ofxTF2Tensor

	// std::cout << "vecFromofxTF2Tensor size: " << vecFromofxTF2Tensor.size() << std::endl;
	// std::cout << "vecFromCppflowTensor size: " << input.get_data<float>().size() << std::endl;

    printVector(vecFromofxTF2Tensor);

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
// 		out.append(std::to_string(vecFromofxTF2Tensor[i]));
//         out.append(" ");
// 	}
// 	ofLog() << out;
// }