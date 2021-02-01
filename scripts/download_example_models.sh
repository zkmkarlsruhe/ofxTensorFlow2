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
URL=https://cloud.zkm.de/index.php/s/gfWEjyEr9X4gyY6/download

SRC=example_models
DEST=../../

##### go

# change to script dir
cd $(dirname "$0")

# download models
curl -LO $URL
unzip download
rm -rf download

cd "$SRC"

# example_basics
unzip model_basic.zip
rm -rf "$DEST"/example_basics/bin/data/model
mv model "$DEST"/example_basics/bin/data

# example_effnet
unzip model_effnet.zip
rm -rf "$DEST"/example_effnet/bin/data/model
mv model "$DEST"/example_effnet/bin/data

# example_keywordspotting
unzip model_keywordspotting.zip
rm -rf "$DEST"/example_keywordspotting/bin/data/model
mv model "$DEST"/example_keywordspotting/bin/data

# example_pix2pix
unzip model_pix2pix_edges2shoes_20epochs.zip
rm -rf "$DEST"/example_pix2pix/bin/data/model
mv model "$DEST"/example_pix2pix/bin/data

# example_style_transfer
unzip models_style_transfer_640x480.zip
rm -rf "$DEST"/example_style_transfer/bin/data/models
mv models "$DEST"/example_style_transfer/bin/data

# cleanup
cd ../
rm -rf "$SRC"
