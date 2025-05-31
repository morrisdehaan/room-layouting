#!/bin/bash

# changeable paramters
SAVE_PATH="results/"
IMG_PATH="images/"
CONFIG_PATH="config/"
START=0
END=100

conda activate svid
python -c "import torch; print(torch.__version__)"

SAVE_LOC=$(python unique_save.py $SAVE_PATH)
echo saving to "$SAVE_LOC"

python instance_seg.py -d $IMG_PATH/ -c $CONFIG_PATH -l $SAVE_LOC
python clip_images.py -d $IMG_PATH -i $SAVE_LOC
python ae_training.py -d $SAVE_LOC
python latent_videos.py -d $SAVE_LOC

# START and END can be added to instance_seg.py and 
# clip_images.py with -s $START and -e $END. 