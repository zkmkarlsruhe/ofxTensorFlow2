#! /bin/sh
#
# Xcode run script for macOS to change tensorflow dylib loader paths and
# install top .app bundle when building
#
# usage: APP
# ie. APP is path/to/OFApp.app
#
# Dan Wilcox ZKM | Hertz-Lab 2021

# stop on error
set -e

# tf version
VER=2.4.0

APP_PATH="$1"
APP_NAME="$(basename ${APP_PATH%.*})"

SRC="$OF_PATH/addons/ofxTensorflow2/libs/tensorflow2/lib/osx"
DEST="$APP_PATH/Contents/Frameworks"

echo "ofxTensorFlow2: install tensorflow libs to $APP_PATH $APP_NAME"

# copy dylibs to app bundle
rsync -aved "$SRC"/*.dylib "$DEST"

# change dylib loader path to Frameworks dir in executable
install_name_tool -change @rpath/libtensorflow.2.dylib @executable_path/../Frameworks/libtensorflow.2.dylib "$APP_PATH/Contents/MacOS/$APP_NAME"
install_name_tool -change @rpath/libtensorflow_framework.2.dylib @executable_path/../Frameworks/libtensorflow_framework.2.dylib "$APP_PATH/Contents/MacOS/$APP_NAME"

# change dylib loader path to Frameworks dir in dylibs
install_name_tool -id @executable_path/../Frameworks/libtensorflow.2.dylib -change @rpath/libtensorflow_framework.2.dylib @executable_path/../Frameworks/libtensorflow_framework.2.dylib "$DEST"/libtensorflow.${VER}.dylib
install_name_tool -id @executable_path/../Frameworks/libtensorflow.2_framework.dylib "$DEST"/libtensorflow_framework.${VER}.dylib
