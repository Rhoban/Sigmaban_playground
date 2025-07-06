#!/bin/bash

if [ $# -le 0 ]
then
    echo "Usage: train.sh [tag]"
    exit
fi

echo "Killing existent python"
killall -9 python3

echo "Running in background..."
OUT_LOG=checkpoints/$i/out.log
nohup uv run playground/sigmaban2024/runner.py \
	--task flat_terrain_backlash \
	--num_timesteps 300000000 \
	--output_dir checkpoints/$1 \
	--wandb \
	> $OUT_LOG &


echo "Showing log"
tail -f $OUT_LOG
