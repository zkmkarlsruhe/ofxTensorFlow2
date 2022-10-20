#! /bin/sh
# configure custom settings in Makefile-based projects
# 1. run this script
# 2. generate projects files with oF ProjectGenerator
# 3. make & run
# Dan Wilcox <dan.wilcox@zkm.de> 2022

# stop on error
set -e

##### variables

PLATFORM="$(uname -s)"

##### commandline

if [ "$1" = "" ] ; then
	echo "usage: path/to/project/dir"
	exit 1
fi

##### main

cd "$1"

if [ ! -e Makefile ] ; then
	echo "Makefile not found, run Project Generator first?"
	exit 1
fi

# add ofxTensorFlow make targets, if missing
if ! grep -q "# ofxTensorFlow2" Makefile ; then
	echo "" >> Makefile
	echo "" >> Makefile
	echo "# ofxTensorFlow2" >> Makefile
	echo "include \$(OF_ROOT)/addons/ofxTensorFlow2/addon_targets.mk" >> Makefile
	echo "Makefile added make targets"
else
	echo "Makefile already configured"
fi
