#!/bin/bash

# Remove any existing container named foundationpose
docker rm -f foundationpose

# Set the current directory
DIR=$(pwd)/

# Allow connections to the X server
xhost +

# Run the Docker container using the modified image
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp \
  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE \
  ddfl0417/foundationpose:ubuntu-22.04 bash -c "cd $DIR && bash"
