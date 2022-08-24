#! /bin/sh
#
# script to clean the models from the example projects
#
# Dan Wilcox ZKM | Hertz-Lab 2021

# stop on error
set -e

DEST=..

##### go

# change to script dir
cd $(dirname "$0")

# example_basics
rm -rfv "$DEST"/example_basics/bin/data/model

# example_efficientnet
rm -rfv "$DEST"/example_basics_efficientnet/bin/data/model

# example_basics_frozen_graph
rm -rfv "$DEST"/example_basics_frozen_graph/bin/data/model.pb

# example_frozen_graph_char_rnn
rm -rfv "$DEST"/example_frozen_graph_char_rnn/bin/data/models

# example_basics_multi_IO
rm -rfv "$DEST"/example_basics_multi_IO/bin/data/model

# example_keyword_spotting
rm -rfv "$DEST"/example_keyword_spotting/bin/data/model

# example_movenet
rm -rfv "$DEST"/example_movenet/bin/data/model

# example_pix2pix
rm -rfv "$DEST"/example_pix2pix/bin/data/model

# example_style_transfer
rm -rfv "$DEST"/example_style_transfer/bin/data/models

# example_style_transfer_arbitrary
rm -rfv "$DEST"/example_style_transfer_arbitrary/bin/data/model

# example_video_matting
rm -rfv "$DEST"/example_video_matting/bin/data/model

# example_yolo_v4
rm -rfv "$DEST"/example_yolo_v4/bin/data/model
rm -rfv "$DEST"/example_yolo_v4/bin/data/dog.jpg
rm -rfv "$DEST"/example_yolo_v4/bin/data/movie.mp4
