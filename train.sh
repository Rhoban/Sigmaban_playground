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

# --restore_checkpoint_path $HOME/Sigmaban_playground/checkpoints/2025_07_05_174512_171704320/ \

echo "Showing log"
tail -f out.log
