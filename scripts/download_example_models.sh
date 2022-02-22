#! /bin/sh
#
# script to download pre-trained models for the example projects,
# downloads all models by default or individual provide zip files by name:
#    download_example_models.sh model_basics
#
# requires: curl, unzip
#
# Dan Wilcox ZKM | Hertz-Lab 2021

# stop on error
set -e

# ZKM NextCloud shared folder direct link
URL=https://cloud.zkm.de/index.php/s/gfWEjyEr9X4gyY6

SRC=example_models
DEST=../..

# known model filenames
MODELS="\
model_basics.zip \
model_basics_multi_IO.zip \
model_basics_frozen_graph.zip \
model_effnet.zip \
models_char_rnn.zip \
model_keywordspotting.zip \
model_multi_pose.zip \
model_pix2pix_edges2shoes_20epochs.zip \
models_style_transfer_640x480.zip \
model_video_matting.zip \
"

# download target(s)
DOWNLOAD="$MODELS"

##### functions

# download from a public NextCloud shared link
# $1: root folder share link URL
# $2: filename
download() {
	local path="download?path=%2F&files=$2"
	RETCODE=$(curl -LO -w "%{http_code}" $URL/$path)
	if [ "$RETCODE" != "200" ] ; then
		echo "download failed: HTTP $RETCODE"
		rm -rf $path
		return
	fi
	mkdir -p $SRC
	mv $path $SRC/$2
}

##### parse command line arguments

HELP="USAGE: $(basename $0) [OPTIONS] [MODELZIP...]

  download pre-trained models for the ofxTensorFlow2 example projects,
  downloads all by default or a list of zip filenames

Options:
  -h,--help    display this help message
  -l,--list    list known model zip filenames
"

while [ "$1" != "" ] ; do
	case $1 in
		-h|--help)
			echo "$HELP"
			exit 0
			;;
		-l|--list)
			for model in $MODELS ; do
				echo "$model"
			done
			exit 0
			;;
		*)
			break
			;;
	esac
	shift 1
done

if [ "$#" -gt 0 ] ; then
	DOWNLOAD="$@"
fi

##### main

# change to script dir
cd $(dirname "$0")

# download targets
for zip in $DOWNLOAD ; do
	download $URL "$zip"
done

cd "$SRC"

# example_basics
if [ -f model_basics.zip ] ; then
	unzip model_basics.zip
	rm -rf "$DEST"/example_basics/bin/data/model
	mv -v model "$DEST"/example_basics/bin/data
fi

# example_efficientnet
if [ -f model_effnet.zip ] ; then
	unzip model_effnet.zip
	rm -rf "$DEST"/example_basics_efficientnet/bin/data/model
	mv -v model "$DEST"/example_basics_efficientnet/bin/data
fi

# example_basics_frozen_graph
if [ -f model_basics_frozen_graph.zip ] ; then
	unzip model_basics_frozen_graph.zip
	rm -rf "$DEST"/example_basics_frozen_graph/bin/data/model.pb
	mv -v model.pb "$DEST"/example_basics_frozen_graph/bin/data
fi

# example_basics_multi_IO
if [ -f model_basics_multi_IO.zip ] ; then
	unzip model_basics_multi_IO.zip
	rm -rf "$DEST"/example_basics_multi_IO/bin/data/model
	mv -v model "$DEST"/example_basics_multi_IO/bin/data
fi

# example_frozen_graph_char_rnn
if [ -f models_char_rnn.zip ] ; then
	unzip models_char_rnn.zip
	rm -rf "$DEST"/example_frozen_graph_char_rnn/bin/data/models
	mv -v models "$DEST"/example_frozen_graph_char_rnn/bin/data
fi

# example_keyword_spotting
if [ -f model_keywordspotting.zip ] ; then
	unzip model_keywordspotting.zip
	rm -rf "$DEST"/example_keyword_spotting/bin/data/model
	mv -v model "$DEST"/example_keyword_spotting/bin/data
fi

# example_movenet
if [ -f model_multi_pose.zip ] ; then
	unzip model_multi_pose.zip
	rm -rf "$DEST"/example_movenet/bin/data/model
	mv -v model "$DEST"/example_movenet/bin/data
fi

# example_pix2pix
if [ -f model_pix2pix_edges2shoes_20epochs.zip ] ; then
	unzip model_pix2pix_edges2shoes_20epochs.zip
	rm -rf "$DEST"/example_pix2pix/bin/data/model
	mv -v model "$DEST"/example_pix2pix/bin/data
fi

# example_style_transfer
if [ -f models_style_transfer_640x480.zip ] ; then
	unzip models_style_transfer_640x480.zip
	rm -rf "$DEST"/example_style_transfer/bin/data/models
	mv -v models "$DEST"/example_style_transfer/bin/data
fi

# example_video_matting
if [ -f model_video_matting.zip ] ; then
	unzip model_video_matting.zip
	rm -rf "$DEST"/example_video_matting/bin/data/model
	rm -rf "$DEST"/example_video_matting/bin/data/codylexi.mp4
	rm -rf "$DEST"/example_video_matting/bin/data/bg.jpg
	mv -v model "$DEST"/example_video_matting/bin/data
	mv -v codylexi.mp4 "$DEST"/example_video_matting/bin/data
	mv -v bg.jpg "$DEST"/example_video_matting/bin/data
fi

# cleanup
cd ../
rm -rf "$SRC"
