#! /bin/sh
#
# script to download pre-built tensorflow C library from
# * official: https://www.tensorflow.org/install/lang_c
# * non-official macOS arm64: https://github.com/vodianyk/libtensorflow-cpu-darwin-arm64
#
# ref: https://stackoverflow.com/a/55434980
#
# Dan Wilcox ZKM | Hertz-Lab 2021-22

# stop on error
set -e

# tf version: optional argument
VER=2.8.0
if [ "$1" != "" ] ; then
	VER=$1
fi

# tf type: cpu or gpu,
# override when running via: TYPE=gpu ./download_tensorflow.sh
if [ "$TYPE" = "" ] ; then
	TYPE=cpu
fi

# default tf download host url
HOST=https://storage.googleapis.com/tensorflow/libtensorflow

# tf download file extension
EXT=tar.gz

# locations
SRC=libtensorflow
DEST=../libs/tensorflow

##### detect

# system detection
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# convert $OS to OF OS naming
OF_OS=unknown
case "$OS" in
	darwin)
		OF_OS=osx
		if [ "$TYPE" = "gpu" ] ; then
			echo "macOS TYPE is cpu-only, switching to cpu"
			TYPE=cpu
		fi
		if [ "$ARCH" = "arm64" ] ; then
			HOST="https://github.com/vodianyk/libtensorflow-cpu-darwin-arm64/raw/main"
			echo "macOS arm64 builds are non-official and not all version are available"
			echo "downloading from https://github.com/vodianyk/libtensorflow-cpu-darwin-arm64"
		fi
		;;
	linux)
		# tf Linux builds are 64 bit only
		if [ "$ARCH" = "x86_64" ] ; then
			OF_OS=linux64
		else
			echo "unsupported architecture: $ARCH"
			exit 1
		fi
		;;
	windowsnt)
		OS=windows
		OF_OS=vs
		EXT=zip
		;;
	mingw* | msys*)
		OS=windows
		OF_OS=msys2
		EXT=zip
		;;
	*)
		echo "unknown or unsupported operating system: $OS"
		exit 1
		;;
esac

# tgz/zip naming is based on system type
DOWNLOAD=libtensorflow-${TYPE}-${OS}-${ARCH}-${VER}

# summary
echo "detected: $OS $ARCH -> $OF_OS"
echo "build type: $TYPE"
echo "downloading: $DOWNLOAD.$EXT"

##### prepare

# change to script dir
cd "$(dirname $0)"

# clear current
rm -rf $DEST/*

##### download & install

# get latest source
RETCODE=$(curl -LO -w "%{http_code}" ${HOST}/${DOWNLOAD}.${EXT})
if [ "$RETCODE" != "200" ] ; then
	echo "download failed: HTTP $RETCODE"
	rm -rf $SRC ${DOWNLOAD}.${EXT}
	exit 1
fi
mkdir -p $SRC
if [ "$EXT" = "zip" ] ; then
	unzip -d $SRC ${DOWNLOAD}.${EXT}
else
	tar -xvf ${DOWNLOAD}.${EXT} -C $SRC
fi

# create dirs
mkdir -p $DEST/lib

# copy licenses
cp -v $SRC/LICENSE $DEST/
cp -v $SRC/THIRD_PARTY_TF_C_LICENSES $DEST/

# copy headers
cp -Rv $SRC/include $DEST/

# copy libs to subdir using OF OS naming so ProjectGenerator finds them
mkdir -p $DEST/lib/${OF_OS}
cp -Rv $SRC/lib/* $DEST/lib/${OF_OS}/

##### cleanup

rm -rf $SRC ${DOWNLOAD}.${EXT}
