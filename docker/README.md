## What is Docker?
Docker provides a way to create containers which have
entirely setup operating system environments along with bundled software.  We
can use a docker image to run commands in a preconfigured and isolated
environment. Any changes that are made inside the image are discarded when the
image stops running, so it is a good way to provide a consistent development
environment.

## Setting up Docker
[Windows](https://docs.docker.com/docker-for-windows/install/) [Mac
OSX](https://docs.docker.com/docker-for-mac/install/)

If you use Linux, Docker is most likely in your distributions software
repository. If you need help, try to google install docker and the name of your
linux

## Getting the Docker Image
The image is currently hosted on Docker Hub, so getting the image is as simple
as running the following command.

```
docker pull bsvh/qubo
```

If you are running Docker on Linux, depending on your configuration you may
need to use elevated privileges. If the above command fails try the following

```
sudo docker pull bsvh/qubo
```

## Using the Image
If you got this far, getting in the environment is as easy as issuing the
following command

```
docker run -it bsvh/qubo bash
```

This should open up a shell in the development environment. If you only want to
issue a single command, you can replace `bash` with the name of the command to
execute. The first thing to notice is that you are running as the root user.
Because docker images are isolated, there is no need to bother with users.

Some notable directors:

* `/catkin_ws` - Directory container the catkin workspace
* `/src/qubo` - Directory containing the source code for qubo. This is pulled
from the github repository when the container image was built, and will most
likely not be up-to-date. You can issue a `git pull` to get the latest source
code, however this will not persist after the image is shutdown. For persistant
storage we will have to use volumes. More on this later.

You can type `exit` in the shell and the image will be shutdown any changes you
made will not be saved.

## Saving your work
To work on files that will not be deleted when the image is shutdown, we need to
create persistant storage. This can be accomplished using volumes. The following
command will create a volume that can be used for persistant storage.

```
docker volume create <name-of-volume>
```

This can then be passed as an argument to the docker run command and the volume
will be mounted to a specified location. We pass the `-v <volume>:<mountpoint>`
switch to docker run to mount the volume.

For example, if we want to create a volume named ram, and to mount it to 
`/src/qubo` in the image, we would issue the following commands.

```
docker volume create ram
docker run -it -v ram:/src/qubo bsvh/qubo bash
```

Any changes you make inside the `/src/qubo` directory will persist after you
exit the image. You can now go inside the `/src/qubo` directory and get the
most up-to-date source code.

```
cd /src/qubo
git clone https://github.com/robotics-at-maryland/qubo .
```

### Linux
If you are running Linux, you are lucky. You can pass a local folder to docker
to use as volume. This can be done in the following manner

```
docker run -it -v <path-to-folder>:<mountpoint-in-docker-image> <cmd>
```

To mount the current directory to `/src/qubo`, you can issue the following

```
docker run -it -v .:/src/qubo bsvh/qubo bash
```
