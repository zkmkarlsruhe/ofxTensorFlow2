# ofxTensorFlow2

This is an openFrameworks addon for TensorFlow 2.
The code has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum).

 Since TensorFlow does not ship a C++ Library we make use of [cppFlow2](https://github.com/serizba/cppflow/tree/cppflow2), which is a C++ wrapper around TensorFlows C API.

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

Also, libtensorflow is provided as pre-compiled dynamic libraries. On macOS these `.dylib` files need to be configured and copied into the build macOS .app. These steps are provided via the `scripts/xcode_install_libs.sh` script and can be invoked when building, either by Xcode or the Makefiles.

#### Xcode build

Enable C++14 features by changing the `CLANG_CXX_LANGUAGE_STANDARD` define in `OF_ROOT/libs/openFrameworksCompiled/project/osx/CoreOF.xcconfig`, for example:

```makefile
CLANG_CXX_LANGUAGE_STANDARD[arch=x86_64] = c++14
```

After generating project files using the OF Project Generator, add the following to one of the Run Script build phases in the Xcode project to invoke the `xcode_install_libs.sh` script:

1. Select the project in the left-hand Xcode project tree
2. Select the project build target under TARGETS
3. Under the Build Phases tab, find the 2nd Run Script, and add the following before the final `echo` line:

```shell
$OF_PATH/addons/ofxTensorflow2/scripts/xcode_install_libs.sh "$TARGET_BUILD_DIR/$PRODUCT_NAME.app";
```

#### Makefile build

Enable C++14 features by changing `-std=c+=11` to `-std=c++14` line 142 in `OF_ROOT/libs/openFrameworksCompiled/project/osx/config.osx.default.mk`:

```makefile
PLATFORM_CXXFLAGS += -std=c++14
```

When building an application using the makefiles, an additional step is required to install & configure the tensorflow2 dylibs into the project .app. This is partially automated by the `scripts/xcode_install_libs.sh` script. To use it, add the following to the Project's `Makefile`:

```makefile
##### ofxTensorflow2

# path to ofxTensorflow2 dylib install script
TF2_INSTALL_SCRIPT=$(OF_ROOT)/addons/ofxTensorflow2/scripts/xcode_install_libs.sh

# install tensorflow2 dylibs into Debug .app
TF2AppDebug: Debug
	OF_PATH=$(OF_ROOT) $(TF2_INSTALL_SCRIPT) `pwd`/bin/$(APPNAME)_debug.app

# install tensorflow2 dylibs into Release app
TF2AppRelease: Release
	OF_PATH=$(OF_ROOT) $(TF2_INSTALL_SCRIPT) `pwd`/bin/$(APPNAME).app
```

This adds two additional targets, one for Debug and the other for Release, which call ther script to install the .dylibs.

For example, to build a debug version of the application *and* install the libs, simply run:

    make TF2AppDebug

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
