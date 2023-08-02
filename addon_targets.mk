##### ofxTensorflow2
#
# note: PLATFORM_LIB_SUBPATH set by
# libs/openFrameworksCompiled/project/makefileCommon/config.share.mk
#

# additional build targets for macOS
ifeq ($(PLATFORM_LIB_SUBPATH),osx)

# path to ofxTensorflow2 dylib install script
TF2_INSTALL_SCRIPT=$(OF_ROOT)/addons/ofxTensorFlow2/scripts/macos_install_libs.sh

# build Debug app and install tensorflow2 dylibs
DebugTF2: Debug
	OF_PATH=$(OF_ROOT) $(TF2_INSTALL_SCRIPT) `pwd`/bin/$(APPNAME)_debug.app

# build Release app and install tensorflow2 dylibs
ReleaseTF2: Release
	OF_PATH=$(OF_ROOT) $(TF2_INSTALL_SCRIPT) `pwd`/bin/$(APPNAME).app

endif

# additional run targets for Linux
ifeq ($(PLATFORM_LIB_SUBPATH),linux64)

# path to local tensorflow shared libraries
TF2_LIBRARY_PATH=$(OF_ROOT)/addons/ofxTensorFlow2/libs/tensorflow/lib/linux64

# build Debug app and run using tensorflow2 lib path
RunDebugTF2: Debug
	export LD_LIBRARY_PATH="$(TF2_LIBRARY_PATH):$$LD_LIBRARY_PATH" && /
	cd bin && ./$(APPNAME)_debug

# build Release app and run using tensorflow2 lib path
RunReleaseTF2: Release
	export LD_LIBRARY_PATH="$(TF2_LIBRARY_PATH):$$LD_LIBRARY_PATH" && /
	cd bin && ./$(APPNAME)

endif
