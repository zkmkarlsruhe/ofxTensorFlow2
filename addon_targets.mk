##### ofxTensorflow2

# additional builds targets for macOS only
ifdef MAC_OS_MIN_VERSION

# path to ofxTensorflow2 dylib install script
TF2_INSTALL_SCRIPT=$(OF_ROOT)/addons/ofxTensorFlow2/scripts/macos_install_libs.sh

# build Debug app and install tensorflow2 dylibs
DebugTF2: Debug
	OF_PATH=$(OF_ROOT) $(TF2_INSTALL_SCRIPT) `pwd`/bin/$(APPNAME)_debug.app

# build Release app and install tensorflow2 dylibs
ReleaseTF2: Release
	OF_PATH=$(OF_ROOT) $(TF2_INSTALL_SCRIPT) `pwd`/bin/$(APPNAME).app

endif
