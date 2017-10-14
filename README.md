# qubo

## Setup Instructions

**Check out our wiki to get started:** <https://github.com/robotics-at-maryland/qubo/wiki>

Project QUBO currently only supports Ubuntu 16.04.  Any other platforms may not (and probably will not) work correctly.

### Compilation

First of all, install all the dependencies in the project by running the handy installation script:
```sh
bash scripts/install_dependencies.bash
```

source the setup script which SHOULD be at the path below, but if you put it somewhere else you'll have to find it yourself. You're going to want to add the source command to your .bashrc file or equivalent, as you'll have to source it every time 

```sh
source /opt/ros/kinetic/setup.bash
```

We use a system called catkin as our build system. to use it cd into qubo/ and call

```sh
catkin_make
```

