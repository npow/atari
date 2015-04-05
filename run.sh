#!/usr/bin/env bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

declare -A ROMS
ROMS[breakout]=5001
ROMS[enduro]=5002
ROMS[pong]=5003
ROMS[qbert]=5004
ROMS[seaquest]=5005
ROMS[montezuma_revenge]=5006

export THEANO_FLAGS=device=gpu0
export ROM=breakout
export ROM_DIR=roms
export RLGLUE_PORT=${ROMS[$ROM]}
export FRAME_SKIP=4
export DISPLAY_SCREEN=true

if [[ $0 -eq "test" ]]; then
export TEST_ARGS="test"
export NETWORK_FILE="$1"
export RLGLUE_PORT=5000
fi

rl_glue &

ale -game_controller rlglue \
  -send_rgb true \
  -restricted_action_set true \
  -frame_skip $FRAME_SKIP \
  -disable_color_averaging true \
  -display_screen $DISPLAY_SCREEN \
  $ROM_DIR/$ROM.bin &

python agent.py $ROM $NETWORK_FILE &

python experiment.py $TEST_ARGS &

wait
