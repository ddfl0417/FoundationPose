#!/bin/bash

DIR=$(pwd)/

docker start foundationpose
docker exec -it foundationpose bash -c "cd $DIR && bash"
