# Attempt to load a config.make file.
# If none is found, project defaults in config.project.make will be used.
ifneq ($(wildcard config.make),)
	include config.make
endif

# make sure the the OF_ROOT location is defined
ifndef OF_ROOT
	OF_ROOT=$(realpath ../../..)
endif

# call the project makefile!
include $(OF_ROOT)/libs/openFrameworksCompiled/project/makefileCommon/compile.project.mk

##### ofxTensorflow2

# path to ofxTensorflow2 dylib install script
TF2_INSTALL_SCRIPT=$(OF_ROOT)/addons/ofxTensorflow2/scripts/xcode_install_libs.sh

# install tensorflow2 dylibs into Debug .app
TF2AppDebug: Debug
	OF_PATH=$(OF_ROOT) $(TF2_INSTALL_SCRIPT) `pwd`/bin/$(APPNAME)_debug.app

# install tensorflow2 dylibs into Release app
TF2AppRelease: Release
	OF_PATH=$(OF_ROOT) $(TF2_INSTALL_SCRIPT) `pwd`/bin/$(APPNAME).app
