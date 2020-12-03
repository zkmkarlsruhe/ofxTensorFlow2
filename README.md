ofxTensorFlow2
=====================================

Introduction
------------
This is an openFrameworks addon for TensorFlow 2.
We make use of [cppFlow2](https://github.com/serizba/cppflow/tree/cppflow2), which is a C++ wrapper around TensorFlows C API. Unfortunately, TensorFlow doesnt ship an C++ Library, because of ABI incompatabilities.

License
-------
State which license you offer your addon under. openFrameworks is distributed under the [MIT License](https://en.wikipedia.org/wiki/MIT_License), and you might consider using this for your repository. By default, `license.md` contains a copy of the MIT license to which you can add your name and the year.

Installation
------------
### Installation
##### Ubuntu with GPU support
- Install Drivers for GPU support (https://www.tensorflow.org/install/gpu)
- Download TensorFlow2 C API (https://www.tensorflow.org/install/lang_c)
- Extract:
  - include/ --> shared_libs/tensorflow2/include
  - lib/ --> shared_libs/tensorflow2/lib/linux64
- add the tensorflow2 lib folder to the LD_LIBRARY_PATH 
```
export LD_LIBRARY_PATH=/home/foo//OFx/addons/ofxTensorFlow2/shared_libs/tensorflow2/lib/linux64/:$LD_LIBRARY_PATH
```
- compile and execute an example 
- if everything works fine you can write the line for editing the LD_LIBRARY_PATH to ~/.bashrc


Dependencies
------------
What other addons are needed to properly work with this one?

Compatibility
------------
Which versions of OF does this addon work with?

Known issues
------------
- some examples might fail if they exceed the GPUs memory but won't report correctly... use "nvidia-smi -l 1" to inspect your GPU

Version history
------------
It make sense to include a version history here (newest releases first), describing new features and changes to the addon. Use [git tags](http://learn.github.com/p/tagging.html) to mark release points in your repo, too!

### Version 0.1 (Date):
Describe relevant changes etc.


