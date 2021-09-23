#! /bin/sh
#
# script to download pre-trained models for the example projects
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

##### functions

# download from a public NextCloud shared link
# $1: root folder share link URL
# $2: filename
download() {
	local path="download?path=%2F&files=$2"
	curl -LO $URL/$path
	mkdir -p $SRC
	mv $path $SRC/$2
}

##### go

# change to script dir
cd $(dirname "$0")

# download models
download $URL model_basics.zip
download $URL model_basics_multi_IO.zip
download $URL model_effnet.zip
download $URL model_keywordspotting.zip
download $URL model_multi_pose.zip
download $URL model_pix2pix_edges2shoes_20epochs.zip
download $URL models_style_transfer_640x480.zip

cd "$SRC"

# example_basics
unzip model_basics.zip
rm -rf "$DEST"/example_basics/bin/data/model
mv model "$DEST"/example_basics/bin/data

# example_basics_multi_IO
unzip model_basics_multi_IO.zip
rm -rf "$DEST"/example_basics_multi_IO/bin/data/model
mv model "$DEST"/example_basics_multi_IO/bin/data

# example_effnet
unzip model_effnet.zip
rm -rf "$DEST"/example_basics_efficientnet/bin/data/model
mv model "$DEST"/example_basics_efficientnet/bin/data

# example_keywordspotting
unzip model_keywordspotting.zip
rm -rf "$DEST"/example_keyword_spotting/bin/data/model
mv model "$DEST"/example_keyword_spotting/bin/data

# example_movenet
unzip model_multi_pose.zip
rm -rf "$DEST"/example_movenet/bin/data/model
mv model "$DEST"/example_movenet/bin/data

# example_pix2pix
unzip model_pix2pix_edges2shoes_20epochs.zip
rm -rf "$DEST"/example_pix2pix/bin/data/model
mv model "$DEST"/example_pix2pix/bin/data

# example_style_transfer
unzip models_style_transfer_640x480.zip
rm -rf "$DEST"/example_style_transfer/bin/data/models
mv models "$DEST"/example_style_transfer/bin/data

# example_video_matting
unzip model_video_matting.zip
rm -rf "$DEST"/example_video_matting/bin/data/model
rm -rf "$DEST"/example_video_matting/bin/data/codylexi.mp4
rm -rf "$DEST"/example_video_matting/bin/data/bg.jpg
mv model "$DEST"/example_video_matting/bin/data
mv codylexi.mp4 "$DEST"/example_video_matting/bin/data
mv bg.jpg "$DEST"/example_video_matting/bin/data

# cleanup
cd ../
rm -rf "$SRC"
