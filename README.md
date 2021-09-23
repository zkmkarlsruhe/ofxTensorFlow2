ofxTensorFlow2
==============

![ofxTensorFlow2 thumbnail](ofxaddons_thumbnail.png)

This is an openFrameworks addon for the TensorFlow 2 ML (Machine Learning) library. The code has been developed by the ZKM | Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum).

Copyright (c) 2021 ZKM | Karlsruhe. 

BSD Simplified License.

For information on usage and redistribution, and for a DISCLAIMER OF ALL
WARRANTIES, see the file, "LICENSE.txt," in this distribution.

Selected examples:

| Style Transfer | Keyword Spotting |
| :--: | :--: 
| ![](media/style_transfer.gif) | ![](media/keyword_spotting.gif) | 

| Pose Estimation | Pix2Pix |
| :--: | :--: |
![](media/movenet.gif) | ![](media/pix2pix.gif) | 

| Video Matting |
| :--: |
| ![](media/video_matting.gif) | 



Description
-----------

ofxTensorFlow2 is an openFrameworks addon for loading and running ML models trained with the TensorFlow 2 ML (Machine Learning) library:

>TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

<https://www.tensorflow.org>

The addon utilizes the TensorFlow 2 C library wrapped by the open source cppflow C++ interface:

>Run TensorFlow models in c++ without Bazel, without TensorFlow installation and without compiling TensorFlow. Perform tensor manipulation, use eager execution and run saved models directly from C++.

<https://github.com/serizba/cppflow/>

Additional classes wrap the process of loading & running a model and utility functions are provided for conversion between common openFrameworks types (images, pixels, audio samples, etc) and TensorFlow2 tensors.

[openFrameworks](http://www.openframeworks.cc) is a cross platform open source toolkit for creative coding in C++.

Quick Start
-----------

Minimal quick start for a Unix shell to clone cppflow, download pre-built TensorFlow 2 dynamic libraries and pre-trained example models, starting in the root openFrameworks folder:

```shell
cd addons
git clone git@hertz-gitlab.zkm.de:Hertz-Lab/Research/intelligent-museum/ofxTensorFlow2.git
cd ofxTensorFlow2
git submodule update --init --recursive
./scripts/download_tensorflow.sh
./scripts/download_example_models.sh
```

For further information, please find detailed instructions below.

_Note: The TensorFlow download script grabs the CPU-optimized build by default._

Requirements
------------

* openFrameworks
* Operating systems:
  - Linux, 64-bit, x86
  - macOS 10.14 (Mojave) or higher, 64-bit, x86
  - Windows, 64-bit x86 (*should* work, not tested)

To use ofxTensorFlow2, first you need to download and install openFrameworks. The examples are developed against the latest release version of openFrameworks on <http://openframeworks.cc/download>.

[OF github repository](https://github.com/openframeworks/openFrameworks)

Currently, ofxTensorFlow2 is being developed on Linux and macOS. Windows *should* work but has not yet been tested.

The main supported operating systems & architectures are those which have pre-built versions of libtensorflow [available for download](https://www.tensorflow.org/install/lang_c) from the TensorFlow website. Other system configurations are possible but may require building and/or installing libtensorflow manually.

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

Since TensorFlow does not ship a C++ Library we make use of [cppflow](https://github.com/serizba/cppflow/), which is a C++ wrapper around the TensorFlow 2 C API.

Pull cppflow to `libs/cppflow` and checkout cppflow:

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

To run applications using ofxTensorFlow2, the path to the addon's `lib/tensorflow` subfolder needs to be added to the `LD_LIBRARY_PATH` environment variable.

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

1. If libtensorflow was downloaded to `libs/tensorflow/`, remove all files in this folder
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

Enable C++14 features by changing `-std=c++11` to `-std=c++14` on line 142 in `OF_ROOT/libs/openFrameworksCompiled/project/osx/config.osx.default.mk`:

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

1. If libtensorflow was downloaded to `libs/tensorflow/`, remove all files in this folder
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

To make this quick, a script is provided to download and install the models for each example (requires a Unix shell, curl, and unzip):

```shell
cd OF_ROOT/addons/ofxTensorFlow2
./scripts/download_example_models.sh
```

By default, the example applications try to load a SavedModel named "model" (or "models" depending on the example) located in `example_XXXX/bin/data/`. When downloading or training a model, please make sure the SavedModel is at this location and has the right name, otherwise update the model load path string.

### Generating Project Files

Project files for the examples are not included so you will need to generate the project files for your operating system and development environment using the OF ProjectGenerator which is included with the openFrameworks distribution.

To (re)generate project files for an *existing* project:

* Click the "Import" button in the ProjectGenerator
* Navigate to the project's parent folder ie. "ofxTensorFlow2", select the base folder for the example project ie. "example_XXXX", and click the Open button
* Click the "Update" button

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

#### Model Format

ofxTensorFlow2 works with the TensorFlow 2 [SavedModel format](https://www.tensorflow.org/guide/saved_model).

When referring to the "SavedModel" we mean the parent folder of the exported neural network containing two subfolder `assets` and `variables` and a `saved_model.pb` file. Do not change anything inside this folder, however renaming the folder is permitted. Keep in mind to use the correct file path within the application.

#### Pretrained Models

Often you don't need or want to train your models from scratch. Therefor, you should take a look at the [TF Hub](tfhub.dev). As TF2 is still rather new, there is not always a SavedModel for your purpose. Besides tfhub.dev you can search GitHub for a TF2 implementation of your model. A great place to start may be [here](https://github.com/Amin-Tgz/awesome-tensorflow-2). If you dont find a pretrained model, it is still easier to run/extend the code of an existing project instead of starting from scratch.

If you happen to find a SavedModel that suits you, but actually don't know the in and output specifications, use the `saved_model_cli` that comes with TensorFlow. For example:
```bash
saved_model_cli show --dir path/to/model/ --tag_set serve --signature_def serving_default
```
should give you the name and expected shape of the in and output tensors. If the names differ from the standard ones or you have more than one in or output tensor you can use `ofxTF2Model::setup()` to specify them. This is also explained in the MultiIO example.

Training Models
---------------

##### Requirements

* Python 3
* Python Package Manger
* Virtual Environments (optional)

##### Recommendations
- [Anaconda](https://docs.anaconda.com/anaconda/install/) / [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (includes all requirements)
- recent [GPU](https://www.tensorflow.org/install/gpu) (10+ series) + software support
- if no GPU is available try free services like [Google"s Colab](https://colab.research.google.com)

We recommend using Python3 as Python2 is not being developed any longer. A python installation is usually extended using a package manager, e.g. pip or conda. To handle the dependencies of Python projects, virtual environments (venvs) are considered best practice. Most beginners to Python use Anaconda or the smaller Miniconda which have all of it to start with.

While you should not mix vens, you can do so for package managers e.g.:

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

With our virtual environment set up and activated we need to install the required python packages. For each example we've listed the required packages using `pip3 freeze > requirements.txt`. You can easily install them by running:

```shell
pip3 install -r requirements.txt
```

As the training procedure and the way of configuring it varies a lot between the examples, please refer to the README.md provided in the `python` folder. Some may require to simply edit a config script and run:

```shell
python3 main.py
```

Some scipts may require to feed additional information to the `main.py` script. 

#### Creating Your Own Project Models

If you want to create your own Deep Learning project, here are some tips to guide you along the way.

##### IDE

Get an __IDE__ (Integrated Development Environment) aka fancy text editor for development. As you will be using Python, choose a specialized IDE, e.g. Spyder (included in Anaconda) or PyCharm. Make sure to set the interpreter of the virtual environment for this project. If you chose to create the virtual environment using conda you will find a subfolder `envs` in the installation folder of anaconda. This includes a folder for every virtual environment. Choose the right one and go to `bin` and select the binary _python_ as interpreter. This way the IDE can run and debug your projects.

##### Python

Get familiar with __Python__. The official [Python tutorial](https://docs.python.org/3/tutorial/index.html) is a great place to start. Python has a lot of functions in its [standard library](https://docs.python.org/3/library/index.html), but there are a lot of other external packages to look out for:

* NumPy (efficient math algorithms and data structures)
* Matplotlib (plotting in the style of Matlab)
* TensorFlow 2 (ML library)

##### Keras

Get familiar with __Keras__. Since TensorFlow 2, [Keras](https://keras.io) is the high level front-end of TensorFlow. It greatly reduces the effort of accessing common data structures (like labeled pictures), defining a neural network architecture and observing the training process using callbacks. Besides that, you can always call TensorFlow's core functions such as data pipelines.

##### Project Structure

Get some __structure__ for your project. Your project could look a little bit like this:

* `data`: stores scripts to download and maybe process some data
* `src`: contains Python code for the model, preprocessing and train, test and eval procedures
* `main.py`: often serves as a front to call the train, eval or test scripts
* `config.py`: stores high level parameters such as learning rate, batch size, etc. Edit this file for different experiments. Formats other than .py are fine too, but it's very easy to integrate. It's a good choice to save this file along with trained models.
* `requirements.txt`: contains required packages

##### Machine Learning

Get familiar with __Machine Learning concepts__. There is plenty of free information out there! Here is a list of material to look into:

* [Coursera](coursera.org): founded by ML expert Andrew Ng, lists free online courses for a lot of fields (including Python and [Machine Learning](https://www.coursera.org/learn/machine-learning))
* [Deeplearning.ai](https://www.deeplearning.ai/): a website dedicated to Deep Learning - also founded by Andrew Ng
* [Deep Learning book](https://www.deeplearningbook.org/): a free website accompanying the book "Deep Learning" by Ian Goodfellow (known for GANs)
* [Stanford CS231](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv): YouTube playlist of Stanford's Computer Vision course CS231
* [Machine Learning Mastery](https://machinelearningmastery.com/): a popular blog about practical ML techniques. It focuses on the ease of use

##### TensorFlow

Get familiar with __TensorFlow's__ [Tutorials](https://www.tensorflow.org/tutorials). Besides learning how to write TensorFlow code, the tutorials will teach you ML concepts like over- and underfitting. Another great place to start is [this repository](https://github.com/Amin-Tgz/awesome-tensorflow-2). It is a vast conglomeration of material related to TensorFlow 2.X.

##### Datasets

Get to know common __datasets__. A great place to start is [Kaggle](kaggle.com). Here you can find thousands of datasets and accompanying code (in form of Python notebooks that run in your browser). [TF datasets](https://www.tensorflow.org/datasets/catalog/overview) is also very popular as most datasets do not require manual download.

##### Inspiration

Get __inspired__ and take the risk of making __errors__! We can not help you with the latter but check out this [repo](https://github.com/vibertthio/awesome-machine-learning-art) for some inspiration.

Developing
----------

You can help develop ofxTensorFlow2 on GitHub: <https://github.com/zkmkalrsruhe/ofxTensorFlow2>

Create an account, clone or fork the repo, then request a push/merge.

If you find any bugs or suggestions please log them to GitHub as well.

Known Issues
------------

### dyld: Library not loaded: @rpath/libtensorflow.2.dylib

On macOS, the libtensorflow dynamic libraries (dylibs) need to be copied into the .app bundle. This error indicates the library loader cannot find the dylibs when the app starts and the build process is missing a step. Please check the "macOS" subsection under the "Installation & Build" section.

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

### Symbol not found: ____chkstk_darwin

The pre-built libtensorflow dynamic libraries downloaded from the TensorFlow website require a minimum of macOS 10.14. On macOS 10.13 or lower, the project may build but will fail on run with a runtime loader error:

~~~
dyld: lazy symbol binding failed: Symbol not found: ____chkstk_darwin
  Referenced from: /Users/na/of_v0.11.0_osx_release/addons/ofxTensorFlow2/example_basics/bin/example_basics.app/Contents/MacOS/./../Frameworks/libtensorflow.2.dylib (which was built for Mac OS X 10.15)
  Expected in: /usr/lib/libSystem.B.dylib
~~~

The only solutions are:

1. upgrade to macOS 10.14 or newer (easier)
2. use libtensorflow compiled for your system:
  * installed to system via a package manager, ie. Homebrew or Macports (harder)
  * or, build libtensorflow manually (probably hardest)

The Intelligent Museum
----------------------

An artistic-curatorial field of experimentation for deep learning and visitor participation

The [ZKM | Center for Art and Media](https://zkm.de/en) and the [Deutsches Museum Nuremberg](https://www.deutsches-museum.de/en/nuernberg/information/) cooperate with the goal of implementing an AI-supported exhibition. Together with researchers and international artists, new AI-based works of art will be realized during the next four years (2020-2023).  They will be embedded in the AI-supported exhibition in both houses. The Project „The Intelligent Museum” is funded by the Digital Culture Programme of the [Kulturstiftung des Bundes](https://www.kulturstiftung-des-bundes.de/en) (German Federal Cultural Foundation) and funded by the [Beauftragte der Bundesregierung für Kultur und Medien](https://www.bundesregierung.de/breg-de/bundesregierung/staatsministerin-fuer-kultur-und-medien) (Federal Government Commissioner for Culture and the Media).

As part of the project, digital curating will be critically examined using various approaches of digital art. Experimenting with new digital aesthetics and forms of expression enables new museum experiences and thus new ways of museum communication and visitor participation. The museum is transformed to a place of experience and critical exchange.

![Logo](media/Logo_ZKM_DMN_KSB.png)
