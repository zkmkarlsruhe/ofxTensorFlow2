#pragma once

#include "cppflow/cppflow.h"
#include <ostream>

class ofxTensor{

    public:

    template <typename T>
    ofxTensor(const T& value) : tensor_(value) {}

    template <typename T>
    ofxTensor(const std::vector<T> & values) : tensor_(values) {}

    template <typename T>
    ofxTensor(const std::vector<T> & values, const std::vector<int64_t>& shape)
        : tensor_(values, shape) {}

    ofxTensor(const cppflow::tensor & tensor) 
        : tensor_(tensor) {}


    template <typename T>
    ofxTensor(const ofPixels & pixels) : tensor_(
            std::vector<T>(pixels.begin(), pixels.end()),
            {pixels.getWidth(), pixels.getHeight(), pixels.getNumChannels()} )
            {}

    ofxTensor(const ofImage & img) : ofxTensor(img.getPixels()) {}

    operator cppflow::tensor & (){
        return tensor_;
    }

    template <typename T>
    std::vector<T> getData(){
        return tensor_.get_data<T>();
    }

    // friend ostream & operator << (ostream & os, const ofxTensor & tensor);
    
    // ostream & operator << (ostream & os){
    //     return os << tensor_;
    // }

    private:

    cppflow::tensor tensor_;

};

// ostream & operator << (ostream & os, const ofxTensor & tensor){
//     return os << tensor.tensor_;
// }