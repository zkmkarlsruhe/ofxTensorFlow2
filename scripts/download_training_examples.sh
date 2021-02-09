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
URL=https://cloud.zkm.de/index.php/s/4STL4aG9NNLC9sD/download
SRC=example_weights
DEST=../..

##### go

# change to script dir
cd $(dirname "$0")

# download models
curl -LO $URL
unzip download
rm -rf download

cd "$SRC"

# model_keywordspotting_h5weights
unzip model_keywordspotting_h5weights.zip
rm -rf "$DEST"/example_keyword_spotting/python/model-attRNN.h5
mv model-attRNN.h5 "$DEST"/example_keyword_spotting/python

# models_style_transfer_checkpoint
unzip models_style_transfer_checkpoint.zip
rm -rf "$DEST"/example_style_transfer/bin/data/models_checkpoint
mv models_checkpoint "$DEST"/example_style_transfer/bin/data

# cleanup
cd ../
rm -rf "$SRC"
