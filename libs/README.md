libs
====

Building libtensorflow for macOS arm64 (Apple Silicon)
------------------------------------------------------

### Dependencies

The following dependencies are required:

* Xcode Commandline Tools
* bazelisk (bazel) build tool

The CLT can be installed on the commandline:

    xcode-select --install

bazelisk can be installed via [Homebrew](https://brew.sh):

    brew install bazelisk

### TL;DR Quickstart

If all dependencies are installed, run the make in this directory to download, build, copy, and clean:

    cd libs
    make tensorflow

... otherwise, each of the steps can be done individually (see following sections).

_The build takes about 30 minutes on an M1 Pro laptop._

On success, clean the build directory to save space:

    make clean

_Note: After a successful build, make sure to run clean otherwise the OF Project Generator will look inside the `libs/build` folder and add the dylibs twice. This will also take a **long time**, so best avoided._

### Download

Download tensorflow via cloning the GitHub repo:

    make tensorflow-download

The sources are downloaded into the `libs/build` temp directory.

### Build

Start building via:

    make tensorflow-build

To override the release version, set the `TF_VER` Makefile variable:

    make tensorflow-build TF_VER=2.16.1

By default, bazel builds for the current architecture. To override, you can set [additional bazel configure options](https://github.com/tensorflow/tensorflow/issues/52160#issuecomment-1053265856) using the `TF_OPTS` Makefile variable:
* to build for x86_64 on arm64 machines: `--cpu=darwin_x86_64`
* to build for arm64 on x86_64 machines: `--cpu=darwin_arm64`

For instance, to build for Intel (x86_64) on a host Apple Silicon machine (arm64):

    make tensorflow-build TF_OPTS="--cpu=darwin_x86_64"

libtensorflow will need to be configured for the platform before building and it does this through a set of questions which are answered via 'y', 'N' or pressing the Enter / Return key. Here are typical answers, basically saying N to AMD, NVIDIA, and mobile device support:
~~~
You have bazel 6.5.0 installed.
Please specify the location of python. [Default is /opt/homebrew/opt/python@3.12/bin/python3.12]:

Found possible Python library paths:
  /opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages
Please input the desired Python library path to use.  Default is [/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages]
ENTER

Do you wish to build TensorFlow with ROCm support? [y/N]: N
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: N
No CUDA support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]:
ENTER

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: N
Not configuring the WORKSPACE for Android builds.

Do you wish to build TensorFlow with iOS support? [y/N]: N
No iOS support will be enabled for TensorFlow.
~~~

### Copy

Once built, the libtensorflow dynamic libraries and required headers can be copied into `libs/tensorflow` with:

    make libtensorflow-copy

### Clean

To clean the build after a successful build and copy, run:

    make tensorflow-clean

### Clobber

To remove copied headers and dynamic libraries, for instance before performing an upgrade to a new version, run:

    make tensorflow-clobber
