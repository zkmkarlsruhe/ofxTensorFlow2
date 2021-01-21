#include "cppflow/cppflow.h"


void testMemoryLayout() {

    ofLog() << "============= Start testing memory layout =============";

    const int size = 16;
    std::vector <int> flat {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

    for (int i=0; i < size; i++){
        std::cout << flat[i] << " ";
    }

    cppflow::tensor tensor(flat, {2,2,2,2});

    auto tensorVector = tensor.get_data<int>();

    for (int i=0; i < size; i++){
        std::cout << tensorVector[i] << " ";
    }


}