ofxTensorFlow2
=====================================

This is an openFrameworks addon for TensorFlow 2.
The code base has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum). 
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


## The Intelligent Museum
An artistic-curatorial field of experimentation for deep learning and visitor participation

The [ZKM | Center for Art and Media](https://zkm.de/en) and the [Deutsches Museum Nuremberg](https://www.deutsches-museum.de/en/nuernberg/information/) cooperate with the goal of implementing an AI-supported exhibition. Together with researchers and international artists, new AI-based works of art will be realized during the next four years (2020-2023).  They will be embedded in the AI-supported exhibition in both houses. The Project „The Intelligent Museum“ is funded by the Digital Culture Programme of the [Kulturstiftung des Bundes](https://www.kulturstiftung-des-bundes.de/en) (German Federal Cultural Foundation).

As part of the project, digital curating will be critically examined using various approaches of digital art. Experimenting with new digital aesthetics and forms of expression enables new museum experiences and thus new ways of museum communication and visitor participation. The museum is transformed to a place of experience and critical exchange.

![Logo](media/Logo_ZKM_DMN_KSB.png)

