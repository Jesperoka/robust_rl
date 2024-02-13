# Docker Quick How-To

## General guide

*Note: robodart specific commands [below](#robodart-specific-commands)*<br><br>

To build docker container (from inside ~/where_my_dockerfile_is/):
```sh
docker build -t name_of_container .
```

To run docker container (from inside ~/where_my_dockerfile_is) and bind a host machine folder `my_folder` to a docker volume:
```sh
docker run -it \
    --mount type=bind,source=absolute/path/to/my_folder,target=/whatever_name_you_want \
    name_of_container:latest 
```

Useful run options:
```sh
    --net=host \
    --privileged \
    --mount type=bind,source=$(pwd)/your-folder-on-host,target=/your-folder-on-vm \
    --device=/dev/ttyUSB0 \
```

To build+run in single command:
```sh
docker run -it \
    $(docker build -q .)
```

## Robodart Specific Commands

#### Setup

1. Put dockerfile in `~/robodart`.

2. Clone `robodart_exploration/` inside `~/robodart/`

3. Build container from dockerfile:
```sh
docker build -t robodart .
```
```sh
xhost +local:docker
```
#### Running

Run the following command from `~/robodart/`:
```sh
docker run -it --net=host --privileged --mount type=bind,source=$(pwd)/robodart_exploration,target=/robodart_exploration robodart:latest
```
```sh
docker run -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix -it --net=host --privileged --mount type=bind,source=$(pwd)/robodart_exploration,target=/robodart_exploration robodart:latest
```

and **while in the docker container** you can navigate to `robodart_exploration`:
```sh
cd robodart_exploration
```
run python files with:
```sh
python3 main.py
```
and to exit the container run:
```sh
exit
```
All changes made to `robodart_exploration`, both within the Docker container and outside the container (on the host machine) are synced. As it stands no other files remain after exiting the container, so if you want to save a file, save it inside `robodart_exploration`.

While `docker run` creates and then starts a container. A stopped container be started with:
```sh
docker start -a -i name_of_container
```
where `name_of_container` can be specified during creation, or they will be pseudo-randomly created and you can run:
```sh
docker ps -a
```
to see a list of all containers, both running and stopped. 

The reason for using `docker start` instead of `docker run` is to save hard drive space, but we can also remove stopped containers with:
```sh
docker container prune
```
Removing containers does not affect volumes.
