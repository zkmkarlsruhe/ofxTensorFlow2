#! /bin/sh
# configure ofxTensorFlow2 custom settings for Xcode project files
# 1. generate projects files with oF ProjectGeneration
# 2. run this script on the project directory (not .xcodeproj)
# 3. open Xcode project and build/run
# Dan Wilcox ZKM | Hertz-Lab 2022

# stop on error
set -e

##### commandline

if [ "$1" = "" ] ; then
	echo "usage: path/to/project/dir"
	exit 1
fi

##### main

cd "$1"

# find first xcode project, there should be only one for oF projects
PROJECT="$(ls -1 | grep xcodeproj | head -n1)"
if [ "$PROJECT" = "" ] ; then
	echo "xcodeproj not found, run ProjectGenerator first?"
	exit 1
fi

# add call ofxTF2 lib script after last line in 2nd run script build phase
# note: if the oF project template ever changes, then LASTLINE likely needs to be updated
if ! grep -q "# ofxTensorFlow2" "$PROJECT/project.pbxproj" ; then
	LASTLINE='# install_name_tool -change @rpath\/libfmod.dylib'
	SCRIPT='$OF_PATH/addons/ofxTensorFlow2/scripts/macos_install_libs.sh "$TARGET_BUILD_DIR/$PRODUCT_NAME.app";'
	printf "\n# ofxTensorFlow2\n${SCRIPT}\n" | sed -i '' "/$LASTLINE/ r /dev/stdin" "$PROJECT/project.pbxproj"
	echo "$PROJECT added macos_install_libs.sh to run script build phase"
else
	echo "$PROJECT already configured"
fi
