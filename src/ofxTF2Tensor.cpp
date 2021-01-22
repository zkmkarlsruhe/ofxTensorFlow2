#include "ofxTF2Model.h"

// ==== constructors ====

ofxTF2Tensor::ofxTF2Tensor(const cppflow::tensor & tensor)
: tensor_(tensor) {}

ofxTF2Tensor::ofxTF2Tensor(const ofImage & img) : ofxTF2Tensor(img.getPixels()) {}

// ==== operators ====

ofxTF2Tensor::operator cppflow::tensor & (){
	return tensor_;
}

bool ofxTF2Tensor::operator == (const cppflow::tensor & rhs) const {

	// check if shapes are the same
	auto lhsShape = tensor_.shape().get_data<shape_t>();
	auto rhsShape = rhs.shape().get_data<shape_t>();
	if (lhsShape != rhsShape) {
		ofLogWarning() << "ofxTF2Tensor: shape mismatch:"
		               << " shape(lhs): " << shapeToString(lhsShape)
		               << " shape(rhs): " << shapeToString(rhsShape);
		return false;
	}

	// check if the data types are the same
	if (tensor_.dtype() != rhs.dtype()){
		ofLogWarning() << "ofxTF2Tensor: dtype mismatch";
		return false;
	}

	return true;
}

std::ostream & operator << (std::ostream & os, const ofxTF2Tensor & tensor){
	return os << tensor.tensor_;
}

// ==== data access ====

std::vector<shape_t> ofxTF2Tensor::getShape() const {
	return tensor_.shape().get_data<shape_t>();
}

cppflow::tensor ofxTF2Tensor::getTensor() const {
	return tensor_;
}

// ==== protected ====

std::string ofxTF2Tensor::shapeToString(const std::vector<shape_t> & shape) const{
	std::string logMSG ("(");
	for (int i = 0; i < shape.size(); i++){
		logMSG.append(std::to_string(shape[i]));
		if (i != shape.size() -1)
			logMSG.append(", ");
	}
	logMSG.append(")");
	return logMSG;
}
