#!/usr/bin/env bash

# Exit on error.
set -e

# Run the docker image with the streaming script. Pass through any arguments.
#  --mount type=bind,source=/home/bongard/Desktop/speeches/,destination=/jetson-inference/speeches/ \
sudo docker run \
  --runtime nvidia \
  -it \
  --rm  \
  --network host \
  --device /dev/snd \
  --mount type=bind,source=/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/,destination=/jetson-inference/power/ \
  whisper-inference \
  python run_inference.py \
  $@
