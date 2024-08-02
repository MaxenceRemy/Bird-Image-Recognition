#!/bin/bash

docker container stop user_api
docker container stop admin_api
docker container stop inference
docker container stop preprocessing

docker container rm user_api
docker container rm admin_api
docker container rm inference
docker container rm preprocessing

# docker volume rm docker_main_volume

docker image prune -a -f

echo "Clean up was successful"
