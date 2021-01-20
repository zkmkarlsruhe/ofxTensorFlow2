# ofxTensorFlow2

This is an openFrameworks addon for TensorFlow 2.
The code has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum).

 Since TensorFlow does not ship a C++ Library we make use of [cppFlow2](https://github.com/serizba/cppflow/tree/cppflow2), which is a C++ wrapper around TensorFlows C API.

## Quick Start

Minimal quick start to clone & download everything needed:

```bash
git clone git@hertz-gitlab.zkm.de:Hertz-Lab/Research/intelligent-museum/ofxTensorFlow2.git
cd ofxTensorFlow2
git submodule update --init --recursive
./scripts/download_tensorflow.sh
```

Detailed instructions follow.

## Installation

Clone (or download and extract) this repository to the addon folder of openframeworks. Replace OF_ROOT with the path to your openFrameworks installation
```bash
cd OF_ROOT/addons
git clone git@hertz-gitlab.zkm.de:Hertz-Lab/Research/intelligent-museum/ofxTensorFlow2.git
```

### Dependencies

* cppflow
* tensorflow2

Pull cppflow to libs/cppflow and checkout cppflow2:
```bash
cd ofxTensorFlow2
git submodule update --init --recursive
```

Download [TensorFlow2 C API](https://www.tensorflow.org/install/lang_c) and extract the following folders to their destination:
  - include/ --> libs/tensorflow2/include
  - lib/ --> libs/tensorflow2/lib/[osx/linux64/msys2/vs]

To make this quick, you can also use a script which automates the download:

    ./scripts/download_tensorflow.sh

### Ubuntu
Add the lib folder to the LD_LIBRARY_PATH. Replace OF_ROOT with the path to your openFrameworks installation.
```bash
export LD_LIBRARY_PATH=OF_ROOT/addons/ofxTensorFlow2/libs/tensorflow2/lib/linux64/:$LD_LIBRARY_PATH
```
[optional] write the previous line to the end of `~/.bashrc` for permanent modification of LD_LIBRARY_PATH

[optional] for GPU support: refer to https://www.tensorflow.org/install/gpu and install driver and packages

### macOS

The cppflow library requires C++14 which needs to be enabled when building on macOS.

libtensorflow is provided as pre-compiled dynamic libraries. On macOS these `.dylib` files need to be configured and copied into the build macOS .app. These steps are automated via the `scripts/macos_install_libs.sh` script and can be invoked when building, either by Xcode or the Makefiles.

#### Xcode build

Enable C++14 features by changing the `CLANG_CXX_LANGUAGE_STANDARD` define in `OF_ROOT/libs/openFrameworksCompiled/project/osx/CoreOF.xcconfig`, for example:

```makefile
CLANG_CXX_LANGUAGE_STANDARD[arch=x86_64] = c++14
```

After generating project files using the OF Project Generator, add the following to one of the Run Script build phases in the Xcode project to invoke the `macos_install_libs.sh` script:

1. Select the project in the left-hand Xcode project tree
2. Select the project build target under TARGETS
3. Under the Build Phases tab, find the 2nd Run Script, and add the following before the final `echo` line:

```shell
$OF_PATH/addons/ofxTensorflow2/scripts/macos_install_libs.sh "$TARGET_BUILD_DIR/$PRODUCT_NAME.app";
```

#### Makefile build

Enable C++14 features by changing `-std=c+=11` to `-std=c++14` line 142 in `OF_ROOT/libs/openFrameworksCompiled/project/osx/config.osx.default.mk`:

```makefile
PLATFORM_CXXFLAGS += -std=c++14
```

When building an application using the makefiles, an additional step is required to install & configure the tensorflow2 dylibs into the project .app. This is partially automated by the `scripts/macos_install_libs.sh` script which is called from the `addon_targets.mk` file. To use it, add the following to the end of the Project's `Makefile`:

```makefile
# ofxTensorflow2
include $(OF_ROOT)/addons/ofxTensorflow2/addon_targets.mk
```

This adds two additional targets, one for Debug and the other for Release, which call the script to install the .dylibs.

For example, to build a debug version of the application *and* install the libs, simply run:

    make DebugTF2

 Similarly, for release builds use:

     make ReleaseTF2

This will also work when building the normal targets using two steps, for example:

    make Debug
    make DebugTF2

## Usage
Each example contains code to create a neural network and export it as SavedModel. Neural networks require training which may take hours or days in order to produce a satisfying output.
Therefor, we provide already trained models. Check the assets of this repository to find the trained models.

When referring to the SavedModel we mean the parent folder of the exported neural network containing two subfolders assets and variables and a saved_model.pb file. Do not change anything inside this folder. However renaming the folder is permited.

By default, the example applications try to load a SavedModel named "model" located in "example_XXXX/bin/data/". When downloading or training a model please make sure the SavedModel is at this location at has the right name (renaming may be necessary).

Afterwards compile and execute the example. For example using make:
```bash
cd example_XXXX
make
make RunRelease
```

**Note**: When creating a new project make sure to copy the Makefile from any example provided and the name of this addon is given in addons.make 


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
python3 main.py
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
