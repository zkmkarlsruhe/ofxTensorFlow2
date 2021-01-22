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
 */


#include "ofMain.h"
#include "ofxTensorFlow2.h"

void testMemoryLayout() {

    ofLog() << "============= Start testing memory layout =============";

    const int size = 16;
    std::vector <int> flat {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

	ofLog() << "testMemoryLayout:" << vectorToString (flat);

    cppflow::tensor tensor(flat, {2,2,2,2});

    auto tensorVector = tensor.get_data<int>();

	ofLog() << "testMemoryLayout:" << vectorToString (tensorVector);


}