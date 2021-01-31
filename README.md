ofxTensorFlow2
==============

![ofxTensorFlow2 thumbnail](ofxaddons_thumbnail.png)

This is an openFrameworks addon for the TensorFlow 2 ML (Machine Learning) library. The code has been developed by the ZKM | Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum).

Copyright (c) 2021 ZKM | Karlsruhe. 

BSD Simplified License.

For information on usage and redistribution, and for a DISCLAIMER OF ALL
WARRANTIES, see the file, "LICENSE.txt," in this distribution.

Description
-----------

ofxTensorFlow2 is an openFrameworks addon for loading and running ML models trained with the TensorFlow 2 ML (Machine Learning) library:

>TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

<https://www.tensorflow.org>

The addon utilizes the TensorFlow 2 C library wrapped by the open source cppflow 2 C++ interface:

>Run TensorFlow models in c++ without Bazel, without TensorFlow installation and without compiling TensorFlow. Perform tensor manipulation, use eager execution and run saved models directly from C++.

<https://github.com/serizba/cppflow/tree/cppflow2>

Additional classes wrap the process of loading & running a model and utility functions are provided for conversion between common openFrameworks types (images, pixels, audio samples, etc) and TensorFlow2 tensors.

[openFrameworks](http://www.openframeworks.cc) is a cross platform open source toolkit for creative coding in C++.

Quick Start
-----------

Minimal quick start to clone cppflow and download pre-built TensorFlow 2 dynamic libraries, starting in the root openFrameworks directory:

```shell
cd addons
git clone git@hertz-gitlab.zkm.de:Hertz-Lab/Research/intelligent-museum/ofxTensorFlow2.git
cd ofxTensorFlow2
git submodule update --init --recursive
./scripts/download_tensorflow.sh
```

To run the example projects, you will need a copy of the pre-trained ML models which you can download as ZIP files, either from the release page on GitHub or from a public shared link here:

<https://cloud.zkm.de/index.php/s/gfWEjyEr9X4gyY6>

Place each "model" or "models" folder into the respective examples' `bin/data` directory.

For further information, please find detailed instructions below.

_Note: The TensorFlow download script grabs the CPU-optimized build by default._

Build Requirements
------------------

To use ofxTensorFlow2, first you need to download and install openFrameworks. The examples are developed against the latest release version of openFrameworks on <http://openframeworks.cc/download>.

[OF github repository](https://github.com/openframeworks/openFrameworks)

Currently, ofxTensorFlow2 is being developed on Linux and macOS. Windows *should* work but has not yet been tested.

Installation and Build
----------------------

Clone (or download and extract) this repository to the addon folder of openFrameworks. Replace OF_ROOT with the path to your openFrameworks installation

```shell
cd OF_ROOT/addons
git clone git@hertz-gitlab.zkm.de:Hertz-Lab/Research/intelligent-museum/ofxTensorFlow2.git
```

### Dependencies

* TensorFlow 2
* cppflow 2

Since TensorFlow does not ship a C++ Library we make use of [cppflow2](https://github.com/serizba/cppflow/tree/cppflow2), which is a C++ wrapper around the TensorFlow 2 C API.

Pull cppflow to `libs/cppflow` and checkout cppflow2:

```shell
cd ofxTensorFlow2
git submodule update --init --recursive
```

Next, download the pre-built [TensorFlow2 C library](https://www.tensorflow.org/install/lang_c) and extract the following folders to their destination:

~~~
include/ --> libs/tensorflow/include
lib/ --> libs/tensorflow/lib/[osx/linux64/msys2/vs]
~~~

To make this quick, you can use a script which automates the download:

```shell
./scripts/download_tensorflow.sh
```
When opting for GPU support set the `TYPE` script variable:

```shell 
TYPE=gpu ./scripts/download_tensorflow.sh
```

See <https://www.tensorflow.org/install/gpu> for more information on GPU support for TensorFlow.

### Ubuntu / Linux

To run applications using ofxTensorFlow2, the path to the addon's `lib/tensorflow` subdirectory needs to be added to the `LD_LIBRARY_PATH` environment variable.

#### Temporary Lib Path Export

The path can be temporarily added via an export on the commandline (replace `OF_ROOT` with the path to your openFrameworks installation) before running the application:

```shell
export LD_LIBRARY_PATH=OF_ROOT/addons/ofxTensorFlow2/libs/tensorflow/lib/linux64/:$LD_LIBRARY_PATH
make run
```

This step can also be automated by additional makefile targets provided by the `addon_targets.mk` file. To use it, add the following to the end of the project's `Makefile`:

```makefile
# ofxTensorFlow2
include $(OF_ROOT)/addons/ofxTensorFlow2/addon_targets.mk
```

This adds two additional targets, one for Debug and the other for Release, which run the application after exporting the `LD_LIBRARY_PATH`. For example, to run a debug version of the application:

    make RunDebugTF2

 Similarly, for release builds use:

    make RunReleaseTF2

#### Permanent Lib Path Export

For a permanent "set and forget" solution, the export line can be added to the end of your shell's user startup script, ie. `~/.zshrc` or `/.bash_profile` to add the path whenever a new shell session is opened. Once set, the manual export is no longer required when running an ofxTensorFlow2 application.

#### Using libtensorflow Installed to the System

To use libtensorflow installed to a system path, ie. by your system's package manager, the path(s) need to be added to the project header include and library search paths and the libraries need to be passed to the linker.

1. If libtensorflow was downloaded to `libs/tensorflow/`, remove all files in this directory
2. Edit `addon_config.mk` under the "linux64" build target: comment the "local path" sections
3. If using the OF Project Generator, (re)regenerate project files for projects using the addon

_Note: When using libtensorflow installed to the system, the `LD_LIBRARY_PATH` export is not needed._

### macOS

The cppflow library requires C++14 which needs to be enabled when building on macOS.

libtensorflow is provided as pre-compiled dynamic libraries. On macOS these `.dylib` files need to be configured and copied into the build macOS .app. These steps are automated via the `scripts/macos_install_libs.sh` script and can be invoked when building, either by Xcode or the Makefiles.

Alternatively, you can use libtensorflow compiled and installed to the system, ie. `/usr/local` or `/usr/opt`. In this case, the dylibs do not need to be copied into the macOS .app, however the built app will not run on other computers without the same libraries installed to the same location.

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
$OF_PATH/addons/ofxTensorFlow2/scripts/macos_install_libs.sh "$TARGET_BUILD_DIR/$PRODUCT_NAME.app";
```

#### Makefile build

Enable C++14 features by changing `-std=c+=11` to `-std=c++14` on line 142 in `OF_ROOT/libs/openFrameworksCompiled/project/osx/config.osx.default.mk`:

```makefile
PLATFORM_CXXFLAGS += -std=c++14
```

When building an application using the makefiles, an additional step is required to install & configure the tensorflow2 dylibs into the project .app. This is partially automated by the `scripts/macos_install_libs.sh` script which is called from the `addon_targets.mk` file. To use it, add the following to the end of the project's `Makefile`:

```makefile
# ofxTensorFlow2
include $(OF_ROOT)/addons/ofxTensorFlow2/addon_targets.mk
```

This adds two additional targets, one for Debug and the other for Release, which call the script to install the .dylibs. For example, to build a debug version of the application *and* install the libs, simply run:

    make DebugTF2

 Similarly, for release builds use:

    make ReleaseTF2

This will also work when building the normal targets using two steps, for example:

    make Debug
    make DebugTF2

#### Using libtensorflow Installed to the System

To use libtensorflow installed to a system path, ie. from a package manager like Homebrew, the path(s) need to be added to the project header include and library search paths and the libraries need to be passed to the linker. The `scripts/macos_install_libs.sh` is not needed.

1. If libtensorflow was downloaded to `libs/tensorflow/`, remove all files in this directory
2. Edit `addon_config.mk` under the "osx" build target:
  * comment the "local path" sections and uncomment the "system path" sections
  * If needed, change the path for your system, ie. `/usr/local` to `/usr/opt` etc
3. If using the OF Project Generator, (re)regenerate project files for projects using the addon

Running the Example Projects
----------------------------

The example projects are located in the `example_XXXX` directories.

### Downloading Pre-Trained Models

Each example contains code to create a neural network and export it in the [SavedModel format](https://www.tensorflow.org/guide/saved_model). Neural networks require training which may take hours or days in order to produce a satisfying output, therefore we provide pre-trained models which you can download as ZIP files, either from the release page on GitHub or from a public shared link here:

<https://cloud.zkm.de/index.php/s/gfWEjyEr9X4gyY6>

Check the assets of this repository to find the a zip for for each example.

By default, the example applications try to load a SavedModel named "model" (or "models" depending on the example) located in `example_XXXX/bin/data/`. When downloading or training a model, please make sure the SavedModel is at this location and has the right name, otherwise update the model load path string.

### Generating Project Files

Project files for the examples are not included so you will need to generate the project files for your operating system and development environment using the OF ProjectGenerator which is included with the openFrameworks distribution.

To (re)generate project files for an *existing* project:

* click the "Import" button in the ProjectGenerator
* navigate the to base folder for the project ie. "luaExample"
* click the "Update" button

If everything went Ok, you should now be able to open the generated project and build/run the example.

### macOS

Open the Xcode project, select the "example_XXXX Debug" scheme, and hit "Run".

For a Makefile build, build and run an example on the terminal:

```shell
cd example_XXXX
make ReleaseTF2
make RunRelease
```

### Linux

For a Makefile build, build and run an example on the terminal:

```shell
cd example_XXXX
make Release
make RunReleaseTF2
```

Create a New ofxTensorFlow2 Project
-----------------------------------

Simply select ofxTensorFlow2 from the available addons in the OF ProjectGenerator before generating a new project. Make sure that all dependencies are installed and downloaded beforehand, otherwise the PG may miss some paths.

Training Models
---------------

_Note: GPU support recommended_

#### Model Format

ofxTensorFlow2 works with the TensorFlow 2 [SavedModel format](https://www.tensorflow.org/guide/saved_model).

When referring to the "SavedModel" we mean the parent folder of the exported neural network containing two subdirectory assets and variables and a `saved_model.pb` file. Do not change anything inside this folder, however renaming the folder is permitted as long as you you change the file path used within the application to match.

#### Requirements

* python3
* [anaconda](https://docs.anaconda.com/anaconda/install/) / [miniconda](https://docs.conda.io/en/latest/miniconda.html) (suggested)

For building a dataset and training a model for use with the ofxTensorFlow2 addon, use Python 3. For ease of use with dependency handling, using virtual environments is recommended. One such tool for this is anaconda or the smaller miniconda.

Install anaconda or miniconda, then install the pip3 package manager using `conda`:

```shell
conda install pip3
```

#### Included Example Projects

For each example project, create a new virtual environment. We will use `conda` to do so:

```shell

cd example_XXXX/python
conda create -n myEnv python=3.7
conda activate myEnv
```
With our virtual environment set up and activated we need to install the required python packages. A common package manager is pip. For each example we've listed the required packages using `pip3 freeze > requirements.txt`. You can easily install them by running:

```shell
pip3 install -r requirements.txt
```

As the training procedure and the way of configuring it varies a lot between the examples, please refer to the README.md provided in the python folder. Some may require to simply edit a config script and run:
```shell
python3 main.py
```
while others may require to feed additional information to the main.oy script. 

#### Creating Your Own Project Models
If you want to create your own Deep Learning project, here are some tips to guide you along the way:
- get an __IDE__. As you will be using Python, choose a specialized IDE, e.g. Spyder (included in Anaconda) or PyCharm. Make sure to set the virtual environment as the interpreter for this project. If you choose to create the virtual environment using conda you will find a directory _envs/_ in the installation folder of anaconda. This directory includes a folder for every virtual environment. Choose the right one and go to _bin/_ and select the binary _python_ as interpreter. This way the IDE can run and debug your projects.
- get familiar with __Python__. The official [Python tutorial](https://docs.python.org/3/tutorial/index.html) is a great place to start. Python has a lot of functions in its [standard library](https://docs.python.org/3/library/index.html), but there are a lot of other external packages to look out for:
  - NumPy (efficient math algorithms and  data structures)
  - Matplotlib (plotting in the style of Matlab)
  - TensorFlow (ML library)
- get familiar with __Keras__. Since TensorFlow v.2 Keras is the high level 
front-end of TensorFlow. It greatly reduces the effort of accessing common data 
structures (like labeled pictures), defining a Neural Network architecture and 
observing the training process using callbacks. Besides that, you can always call TensorFlow's core functions such as data pipelines.
- get some __structure__ for your project. Your project could look a little bit like this:
  - data/: stores scripts to download and maybe process some data
  - src/: contains python code for the model, preprocessing and train, test and 
eval procedures.
  - main.py: often serves as a front to call the train, eval or test scripts
  - config.py: stores high level parameters such as learning rate, batch size, etc.
Edit this file for different experiments. Other formats than .py are fine too, 
but it's very easy to integrate. It's a good choice to save this file along with 
trained models.
  - requirements.txt: contains required packages
- get familiar with __Machine Learning concepts__. There is plenty of free information out there! Here is a list of material to look into:
  - [Coursera](coursera.org): created by ML expert Andrew Ng, lists free online courses for a lot of fields (including [Machine Learning](https://www.coursera.org/learn/machine-learning))
  - [Stanford CS231](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv): YouTube playlist of Stanford's Computer Vision course CS231.
  - [Machine Learning Mastery](https://machinelearningmastery.com/): a popular blog about ML techniques. It focuses on the practical use.
- get familiar with __TensorFlow's__ [__Tutorials__](https://www.tensorflow.org/tutorials). Besides learning how to write TensorFlow code, the tutorials will teach you ML concepts like over- and underfitting.
- get to know common __datasets__. A great place to start is [Kaggle](kaggle.com). Here you can find thousands of datasets and accompanying code (in form of pyhton notebooks that run in your browser).
- get __inspired__ and risk making __errors__! We can not help you with the latter but checkout this [repo](https://github.com/vibertthio/awesome-machine-learning-art) for some inspiration.


Developing
----------

You can help develop ofxTensorFlow2 on GitHub: <https://github.com/zkmkalrsruhe/ofxTensorFlow2>

Create an account, clone or fork the repo, then request a push/merge.

If you find any bugs or suggestions please log them to GitHub as well.

Known Issues
------------

### EXC_BAD_INSTRUCTION Crash on macOS

The pre-built libtensorflow downloaded to `libs/tensorflow` comes with AVX (Advanced Vector Extensions) enabled which is an extension to the Intel x86 instruction set for fast vector math. CPUs older than circa 2013 may not support this and the application will simply crash with error such as:

~~~
in libtensorflow_framework.2.dylib
...
EXC_BAD_INSTRUCTION (code=EXC_I386_INVOP, subcode=0x0)
~~~

This problem may also be seen when using libtensorflow installed via Homebrew.

The only solution is to build libtensorflow from source with AVX disabled use a machine with a newer CPU. To check if your CPU supports AVX use:
```shell
# print all CPU features
sysctl -a | grep cpu.feat

# prints only if CPU supports AVX
sysctl -a | grep cpu.feat | grep AVX
```

Systems confirmed: Mac Pro (Mid 2012)

The Intelligent Museum
----------------------

An artistic-curatorial field of experimentation for deep learning and visitor participation

The [ZKM | Center for Art and Media](https://zkm.de/en) and the [Deutsches Museum Nuremberg](https://www.deutsches-museum.de/en/nuernberg/information/) cooperate with the goal of implementing an AI-supported exhibition. Together with researchers and international artists, new AI-based works of art will be realized during the next four years (2020-2023).  They will be embedded in the AI-supported exhibition in both houses. The Project „The Intelligent Museum“ is funded by the Digital Culture Programme of the [Kulturstiftung des Bundes](https://www.kulturstiftung-des-bundes.de/en) (German Federal Cultural Foundation).

As part of the project, digital curating will be critically examined using various approaches of digital art. Experimenting with new digital aesthetics and forms of expression enables new museum experiences and thus new ways of museum communication and visitor participation. The museum is transformed to a place of experience and critical exchange.

![Logo](media/Logo_ZKM_DMN_KSB.png)
