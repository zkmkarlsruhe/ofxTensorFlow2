# ofxTensorFlow2

This is an openFrameworks addon for TensorFlow 2.
The code has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum).

 Since TensorFlow does not ship a C++ Library we make use of [cppFlow2](https://github.com/serizba/cppflow/tree/cppflow2), which is a C++ wrapper around TensorFlows C API.


## Installation
Clone (or download and extract) this repository to the addon folder of openframeworks.
```bash
cd OF_ROOT/addons
git clone git@hertz-gitlab.zkm.de:Hertz-Lab/Research/intelligent-museum/ofxTensorFlow2.git
```
Pull the third party library cppflow->cppflow2
```bash
cd ofxTensorFlow2
git submodule update --init --recursive
```
Download [TensorFlow2 C API](https://www.tensorflow.org/install/lang_c). Then extract the following folders to their destination:
  - include/ --> libs/tensorflow2/include
  - lib/ --> libs/tensorflow2/lib/[macOS/linux64]



### Ubuntu
Add the lib folder to the LD_LIBRARY_PATH (replace OF_ROOT with the full path to the ofx installation)
```bash
export LD_LIBRARY_PATH=OF_ROOT/addons/ofxTensorFlow2/libs/tensorflow2/lib/linux64/:$LD_LIBRARY_PATH
```
[optional] write the previous line to the end of ~/.bashrc for permanent modification of LD_LIBRARY_PATH

[optional] for GPU support: refer to https://www.tensorflow.org/install/gpu and install driver and packages

### macOS
Enable c++14 features: Change line 132 in OF_ROOT/libs/openFrameworksCompiled/project/osx/config.osx.default.mk:
```bash
PLATFORM_CXXFLAGS += -std=c++14
```


## Usage
Each example contains code to train neural networks and export them as .pb files (SavedModel). However, we will provide already trained models.

Models need to be placed in example_XXXX/bin/data.

Afterwards compile and execute the example.
```bash
cd example_XXXX
make
make RunRelease
```


## Training (GPU support recommended)
#### Requirements
python3: for ease of use install [anaconda](https://docs.anaconda.com/anaconda/install/) or the smaller [miniconda](https://docs.conda.io/en/latest/miniconda.html) (both include conda which we will use to create virtual environments, however only python required)

pip3
```bash
conda install pip
```

#### Execution
For each example create a new virtual environment. We will use conda to do so:
```bash
cd example_XXXX/python
conda create -n myEnv python=3.7
conda activate myEnv
```
Install required python packages
```bash
pip3 install -r requirements.txt
```
Run the python script to start training
```bash
python3 train.py
```
To configure the training process refer to the README of each example.


## Known issues
please take a look at the [issues](https://hertz-gitlab.zkm.de/Hertz-Lab/Research/intelligent-museum/ofxTensorFlow2/-/issues?scope=all&utf8=%E2%9C%93&state=all)


## License
[MIT License](https://en.wikipedia.org/wiki/MIT_License)


## The Intelligent Museum
An artistic-curatorial field of experimentation for deep learning and visitor participation

The [ZKM | Center for Art and Media](https://zkm.de/en) and the [Deutsches Museum Nuremberg](https://www.deutsches-museum.de/en/nuernberg/information/) cooperate with the goal of implementing an AI-supported exhibition. Together with researchers and international artists, new AI-based works of art will be realized during the next four years (2020-2023).  They will be embedded in the AI-supported exhibition in both houses. The Project „The Intelligent Museum“ is funded by the Digital Culture Programme of the [Kulturstiftung des Bundes](https://www.kulturstiftung-des-bundes.de/en) (German Federal Cultural Foundation).

As part of the project, digital curating will be critically examined using various approaches of digital art. Experimenting with new digital aesthetics and forms of expression enables new museum experiences and thus new ways of museum communication and visitor participation. The museum is transformed to a place of experience and critical exchange.

![Logo](media/Logo_ZKM_DMN_KSB.png)
