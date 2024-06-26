#! /bin/sh
if [ "$1" = "" ] ; then
	echo "usage: path/to/repo/tensorflow [major|minor|patch]"
	exit 1
fi
TF_VERSION_H="$1/tensorflow/core/public/version.h"
if [ ! -f "$TF_VERSION_H" ] ; then
	echo "$TF_VERSION_H not found"
	exit 1
fi
MAJOR=$(grep "^\s*#\s*define\s*TF_MAJOR_VERSION\>" "$TF_VERSION_H" | \
    sed 's|^.define *TF_MAJOR_VERSION *\([0-9]*\).*|\1|')
MINOR=$(grep "^\s*#\s*define\s*TF_MINOR_VERSION\>" "$TF_VERSION_H" | \
    sed 's|^.define *TF_MINOR_VERSION *\([0-9]*\).*|\1|')
PATCH=$(grep "^\s*#\s*define\s*TF_PATCH_VERSION\>" "$TF_VERSION_H" | \
    sed 's|^.define *TF_PATCH_VERSION *\([0-9]*\).*|\1|')
case "$2" in
	major) echo "$MAJOR" ;;
	minor) echo "$MINOR" ;;
	patch) echo "$PATCH" ;;
	*) echo "$MAJOR.$MINOR.$PATCH" ;;
esac
