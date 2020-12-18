ofxTensorFlow2
=====================================

This is an openFrameworks addon for TensorFlow 2.
The code has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum).

 Since TensorFlow does not ship a C++ Library - allegedly because of ABI incompatibilities -  we make use of [cppFlow2](https://github.com/serizba/cppflow/tree/cppflow2), which is a C++ wrapper around TensorFlows C API.


### License
[MIT License](https://en.wikipedia.org/wiki/MIT_License)


### Installation
- Pull the third party library cppflow->cppflow2
```bash
git submodule update --init --recursive
```
- Download [TensorFlow2 C API](https://www.tensorflow.org/install/lang_c)
- Extract the following folder to their destination:
  - include/ --> shared_libs/tensorflow2/include
  - lib/ --> shared_libs/tensorflow2/lib/[macOS/linux64]

#### Ubuntu
- add the lib folder to the LD_LIBRARY_PATH (replace OF_ROOT with the full path to the ofx installation)
```bash
export LD_LIBRARY_PATH=OF_ROOT/addons/ofxTensorFlow2/shared_libs/tensorflow2/lib/linux64/:$LD_LIBRARY_PATH
```
- [optional] write the previous line to ~/.bashrc for permanent modification of LD_LIBRARY_PATH

- [optional] for GPU support: refer to https://www.tensorflow.org/install/gpu and install driver and packages

#### macOS
- enable c++14 features: Change line 132 in OF_ROOT/libs/openFrameworksCompiled/project/osx/config.osx.default.mk:
```bash
PLATFORM_CXXFLAGS += -std=c++14
```


### Usage
run the python script to create the model (requires a TensorFlow2 python installation, e.g. in an virtual environment)
```bash
cd example_XXXX/python
conda create -n myEnv python=3.7
conda activate myEnv
pip3 install -r requirements.txt
python3 main.py
```
now that we have a TensorFlow SavedModel, compile the ofxExample:
```bash
cd ..
make
make RunRelease
```


### Known issues
please take a look at the [issues](https://hertz-gitlab.zkm.de/Hertz-Lab/Research/intelligent-museum/ofxTensorFlow2/-/issues?scope=all&utf8=%E2%9C%93&state=all)

## The Intelligent Museum
An artistic-curatorial field of experimentation for deep learning and visitor participation

The [ZKM | Center for Art and Media](https://zkm.de/en) and the [Deutsches Museum Nuremberg](https://www.deutsches-museum.de/en/nuernberg/information/) cooperate with the goal of implementing an AI-supported exhibition. Together with researchers and international artists, new AI-based works of art will be realized during the next four years (2020-2023).  They will be embedded in the AI-supported exhibition in both houses. The Project „The Intelligent Museum“ is funded by the Digital Culture Programme of the [Kulturstiftung des Bundes](https://www.kulturstiftung-des-bundes.de/en) (German Federal Cultural Foundation).

As part of the project, digital curating will be critically examined using various approaches of digital art. Experimenting with new digital aesthetics and forms of expression enables new museum experiences and thus new ways of museum communication and visitor participation. The museum is transformed to a place of experience and critical exchange.

![Logo](media/Logo_ZKM_DMN_KSB.png)
