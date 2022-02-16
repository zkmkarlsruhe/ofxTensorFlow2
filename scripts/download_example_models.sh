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

# download target
DOWNLOAD="all"
if [ "$1" != "" ] ; then
	DOWNLOAD=$1
fi

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

##### go

# change to script dir
cd $(dirname "$0")

# download models
case "$DOWNLOAD" in
	all)
		echo "ka"
		download $URL model_basics.zip
		download $URL model_basics_multi_IO.zip
		download $URL model_basics_frozen_graph.zip
		download $URL model_effnet.zip
		download $URL model_keywordspotting.zip
		download $URL model_multi_pose.zip
		download $URL model_pix2pix_edges2shoes_20epochs.zip
		download $URL models_style_transfer_640x480.zip
		download $URL model_video_matting.zip
		;;
	*)
		echo "lala"
		download $URL ${DOWNLOAD}.zip
		;;
esac

cd "$SRC"

# example_basics
if [ -f model_basics.zip ] ; then
	unzip model_basics.zip
	rm -rf "$DEST"/example_basics/bin/data/model
	mv model "$DEST"/example_basics/bin/data
fi

# example_basics_frozen_graph
if [ -f model_basics_frozen_graph.zip ] ; then
	unzip model_basics_frozen_graph.zip
	rm -rf "$DEST"/example_basics_frozen_graph/bin/data/model.pb
	mv model.pb "$DEST"/example_basics_frozen_graph/bin/data
fi

# example_basics_multi_IO
if [ -f model_basics_multi_IO.zip ] ; then
	unzip model_basics_multi_IO.zip
	rm -rf "$DEST"/example_basics_multi_IO/bin/data/model
	mv model "$DEST"/example_basics_multi_IO/bin/data
fi

# example_effnet
if [ -f model_effnet.zip ] ; then
	unzip model_effnet.zip
	rm -rf "$DEST"/example_basics_efficientnet/bin/data/model
	mv model "$DEST"/example_basics_efficientnet/bin/data
fi

# example_keywordspotting
if [ -f model_keywordspotting.zip ] ; then
	unzip model_keywordspotting.zip
	rm -rf "$DEST"/example_keyword_spotting/bin/data/model
	mv model "$DEST"/example_keyword_spotting/bin/data
fi

# example_movenet
if [ -f model_multi_pose.zip ] ; then
	unzip model_multi_pose.zip
	rm -rf "$DEST"/example_movenet/bin/data/model
	mv model "$DEST"/example_movenet/bin/data
fi

# example_pix2pix
if [ -f model_pix2pix_edges2shoes_20epochs.zip ] ; then
	unzip model_pix2pix_edges2shoes_20epochs.zip
	rm -rf "$DEST"/example_pix2pix/bin/data/model
	mv model "$DEST"/example_pix2pix/bin/data
fi

# example_style_transfer
if [ -f models_style_transfer_640x480.zip ] ; then
	unzip models_style_transfer_640x480.zip
	rm -rf "$DEST"/example_style_transfer/bin/data/models
	mv models "$DEST"/example_style_transfer/bin/data
fi

# example_video_matting
if [ -f model_video_matting.zip ] ; then
	unzip model_video_matting.zip
	rm -rf "$DEST"/example_video_matting/bin/data/model
	rm -rf "$DEST"/example_video_matting/bin/data/codylexi.mp4
	rm -rf "$DEST"/example_video_matting/bin/data/bg.jpg
	mv model "$DEST"/example_video_matting/bin/data
	mv codylexi.mp4 "$DEST"/example_video_matting/bin/data
	mv bg.jpg "$DEST"/example_video_matting/bin/data
fi

# cleanup
cd ../
rm -rf "$SRC"
