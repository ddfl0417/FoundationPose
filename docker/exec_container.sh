#!/bin/bash

# Set the current directory
DIR=$(pwd)/

# Allow connections to the X server
xhost +

docker start foundationpose
docker exec -it foundationpose bash -c "cd $DIR && bash"