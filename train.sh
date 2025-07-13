#!/bin/bash

if [ $# -le 0 ]
then
    echo "Usage: train.sh [tag]"
    exit
fi

echo "Killing existent python"
killall -9 python3

echo "Running in background..."
CHECKPOINT_DIR=checkpoints/$1
mkdir -p $CHECKPOINT_DIR
OUT_LOG=$CHECKPOINT_DIR/out.log
touch $OUT_LOG
nohup uv run playground/sigmaban2024/runner.py \
	--task flat_terrain_backlash_parkour \
	--env parkour \
	--num_timesteps 300000000 \
	--output_dir $CHECKPOINT_DIR \
	--wandb \
	> $OUT_LOG &

sleep 3
echo "Showing log"
tail -f $OUT_LOG
