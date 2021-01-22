#pragma once

#include "cppflow/cppflow.h"


// todo NEED static method that returns the data type
// todo << operator not working
// todo test image and pixel inputs
// todo move shapeToString to ofxTensorFlow2Utils.h
// todo write and test image and pixel outputs
// 

// a shape is represented as a vector of shape_t
typedef int32_t shape_t;

// a simple wrapper for cppflow::tensor
// note: getData<T> (and cppflow::tensor.get_data<T>) does not convert the value
class ofxTF2Tensor{

    public:

    // ==== Constructors ====
    // forwarding to cppflow constructors
    template <typename T>
    ofxTF2Tensor(const T& value) 
        : tensor_(value) {}

    template <typename T>
    ofxTF2Tensor(const std::vector<T> & values) 
        : tensor_(values) {}

    template <typename T>
    ofxTF2Tensor(const std::vector<T> & values, const std::vector<int64_t>& shape)
        : tensor_(values, shape) {}

    ofxTF2Tensor(const cppflow::tensor & tensor) 
        : tensor_(tensor) {}

    // constuctors for openframework interaction
    template <typename T>
    ofxTF2Tensor(const ofPixels & pixels) : tensor_(
            std::vector<T>(pixels.begin(), pixels.end()),
            {pixels.getWidth(), pixels.getHeight(), pixels.getNumChannels()} )
            {}

    ofxTF2Tensor(const ofImage & img) : ofxTF2Tensor(img.getPixels()) {}


    // ==== operators ====

    // implicit cast to cppflow::tensor 
    // especially useful for using cppflow ops with ofxTF2Tensor
    operator cppflow::tensor & (){
        return tensor_;
    }

    // check if tensors are comparable
    bool operator == (const cppflow::tensor & rhs) const {

        // check if shapes are the same 
        auto lhsShape = tensor_.shape().get_data<shape_t>();
        auto rhsShape = rhs.shape().get_data<shape_t>();
        if ( lhsShape != rhsShape) {
            ofLog() << "ofxTF2Tensor: shape mismatch:"
                    << " shape(lhs): " << shapeToString(lhsShape)
                    << " shape(rhs): " << shapeToString(rhsShape);
            return false;
        }

        // check if the data types are the same
        if (tensor_.dtype() != rhs.dtype()){
            ofLog() << "ofxTF2Tensor: dtype mismatch";
            return false;
        }

        return true;
    }

    // friend ostream & operator << (ostream & os, const ofxTF2Tensor & tensor);
    // ostream & operator << (ostream & os){
    //     return os << "works";
    // }

    // check if both tensors share the same values
    template <typename T>
    bool equals (const cppflow::tensor & rhs) const {

        if (!( *this == rhs )){
            ofLog() << "ofxTF2Tensor: tensors not comparable";
            return false;
        }
        if ( tensor_.get_data<T>() != rhs.get_data<T>() ) {
            ofLog() << "ofxTF2Tensor: value mismatch";
            return false;
        }
        return true;
    }

    std::vector<shape_t> getShape() const {
        return tensor_.shape().get_data<shape_t>();
    }
    cppflow::tensor getTensor() const {
        return tensor_;
    }
    // this function will not convert the values. Make sure to use the correct 
    // type. That is, when using int as template parameter on a float tensor
    // values will not get truncated but the bytes will be interpreted as int. 
    template <typename T>
    std::vector<T> getVector() const {
        return tensor_.get_data<T>();
    }


    private:

    std::string shapeToString(const std::vector<shape_t> & shape) const{
        std::string logMSG ("(");
        for (int i=0; i < shape.size(); i++){
            logMSG.append(std::to_string(shape[i]));
            if (i != shape.size() -1)
                logMSG.append(", ");
        }
        logMSG.append(")");
        return logMSG;
    }


    cppflow::tensor tensor_;

};

// ostream & operator << (ostream & os, const ofxTF2Tensor & tensor){
//     return os << tensor.tensor_;
// }
