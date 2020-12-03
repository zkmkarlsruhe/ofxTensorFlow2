ofxTensorFlow2
=====================================

This is an openFrameworks addon for TensorFlow 2.
We make use of [cppFlow2](https://github.com/serizba/cppflow/tree/cppflow2), which is a C++ wrapper around TensorFlows C API. Unfortunately, TensorFlow doesnt ship an C++ Library, because of ABI incompatabilities.

### License
[MIT License](https://en.wikipedia.org/wiki/MIT_License)

### Installation
##### Ubuntu with GPU support
- Install driver and packages for GPU support (https://www.tensorflow.org/install/gpu)
- Pull the third party library cppflow->cppflow2
```
git submodule update --init --recursive
```
- Download TensorFlow2 C API (https://www.tensorflow.org/install/lang_c)
- Extract the following folder to their destination:
  - include/ --> shared_libs/tensorflow2/include
  - lib/ --> shared_libs/tensorflow2/lib/linux64
- add the tensorflow2/ folder to the LD_LIBRARY_PATH 
```
export LD_LIBRARY_PATH=/home/foo/OFx/addons/ofxTensorFlow2/shared_libs/tensorflow2/lib/linux64/:$LD_LIBRARY_PATH
```

### Usage
- compile and execute an example
```
cd example_basic
make
```
- run the python script to create the model (requires a tensorflow2 python installation)
```
conda activate myEnv
python3 create_model.py
```

### Next
- if everything works fine you can write the line for editing the LD_LIBRARY_PATH to ~/.bashrc


### Compatibility
Which versions of OF does this addon work with?

### Known issues
- some examples might fail if they exceed the GPUs memory but won't report correctly... use "nvidia-smi -l 1" to inspect your GPU

