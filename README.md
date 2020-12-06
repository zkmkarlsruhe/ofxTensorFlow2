ofxTensorFlow2
=====================================

This is an openFrameworks addon for TensorFlow 2.
The code base has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum).
We make use of [cppFlow2](https://github.com/serizba/cppflow/tree/cppflow2), which is a C++ wrapper around TensorFlows C API. Unfortunately, TensorFlow does not ship a C++ Library, because of ABI incompatibilities.


### License
[MIT License](https://en.wikipedia.org/wiki/MIT_License)


### Installation
- Pull the third party library cppflow->cppflow2
```
git submodule update --init --recursive
```
- Download TensorFlow2 C API (https://www.tensorflow.org/install/lang_c)
- Extract the following folder to their destination:
  - include/ --> shared_libs/tensorflow2/include
  - lib/ --> shared_libs/tensorflow2/lib/[macOS/linux64]

##### Ubuntu
- add the lib folder to the LD_LIBRARY_PATH
```
export LD_LIBRARY_PATH=OF_ROOT/addons/ofxTensorFlow2/shared_libs/tensorflow2/lib/linux64/:$LD_LIBRARY_PATH
```
- write the previous line to ~/.bashrc for permanent modification of LD_LIBRARY_PATH
- modify the linker flag: insert your lib install path the following line in addon_config.mk:
```
ADDON_LDFLAGS = -L OF_ROOT/addons/ofxTensorFlow2/shared_libs/tensorflow2/lib/linux64
```
for GPU support
- Refer to https://www.tensorflow.org/install/gpu and install driver and packages

##### macOS
- modify the linker flag: insert your lib install path the following line in addon_config.mk:
```
ADDON_LDFLAGS = OF_ROOT/addons/ofxTensorFlow2/shared_libs/tensorflow2/lib/macOS/libtensorflow.dylib
```
- enable c++14 features. Change line 132 in OF_ROOT/libs/openFrameworksCompiled/project/osx/config.osx.default.mk:
```
PLATFORM_CXXFLAGS += -std=c++14
```


### Usage
- compile and execute an example
```
cd example_basic
make
```
- run the python script to create the model (requires a TensorFlow2 python installation)
```
conda activate myEnv
python3 create_model.py
```


### Known issues
- some examples might fail if they exceed the GPUs memory but won't report correctly... use "nvidia-smi -l 1" to inspect your GPU


## The Intelligent Museum
An artistic-curatorial field of experimentation for deep learning and visitor participation

The [ZKM | Center for Art and Media](https://zkm.de/en) and the [Deutsches Museum Nuremberg](https://www.deutsches-museum.de/en/nuernberg/information/) cooperate with the goal of implementing an AI-supported exhibition. Together with researchers and international artists, new AI-based works of art will be realized during the next four years (2020-2023).  They will be embedded in the AI-supported exhibition in both houses. The Project „The Intelligent Museum“ is funded by the Digital Culture Programme of the [Kulturstiftung des Bundes](https://www.kulturstiftung-des-bundes.de/en) (German Federal Cultural Foundation).

As part of the project, digital curating will be critically examined using various approaches of digital art. Experimenting with new digital aesthetics and forms of expression enables new museum experiences and thus new ways of museum communication and visitor participation. The museum is transformed to a place of experience and critical exchange.

![Logo](media/Logo_ZKM_DMN_KSB.png)
