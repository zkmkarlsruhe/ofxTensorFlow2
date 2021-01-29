#! /bin/sh
#
# script to download pre-built tensorflow C library from
# https://www.tensorflow.org/install/lang_c
#
# Dan Wilcox ZKM | Hertz-Lab 2021

# stop on error
set -e

# tf version
VER=2.4.0

# tf type: cpu or gpu,
# override when running via: TYPE=gpu ./download_tensorflow.sh
if [ "$TYPE" = "" ] ; then
	TYPE=cpu
fi

# tf download host url
HOST=https://storage.googleapis.com/tensorflow/libtensorflow

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
		;;
	mingw* | msys*)
		OS=windows
		OF_OS=msys2
		;;
	*)
		echo "unknown or unsupported operating system: $OS"
		exit 1
		;;
esac

# tgz naming is based on system type
TGZ=libtensorflow-${TYPE}-${OS}-${ARCH}-${VER}.tar.gz

# summary
echo "detected: $OS $ARCH -> $OF_OS"
echo "build type: $TYPE"
echo "downloading: $TGZ"

##### prepare

# change to script dir
cd "$(dirname $0)"

# clear current
rm -rf $DEST/*

##### download & install

# get latest source
curl -O ${HOST}/${TGZ}
mkdir -p $SRC
tar -xvf $TGZ -C $SRC

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

rm -rf $SRC $TGZ
