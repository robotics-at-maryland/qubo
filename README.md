# qubo

## Setup Instructions

Project QUBO currently only supports Ubuntu 14.04.  Any other platforms may not (and probably will not) work correctly.

### Compilation

First of all, install all the dependencies in the project by running the handy installation script:
```sh
bash scripts/install_dependencies.bash
```

source the setup script which SHOULD be at the path below, but if you put it somewhere else you'll have to find it yourself. You're going to want to add the source command to your .bashrc file or equivalent, as you'll have to source it every time 

```sh
source /opt/ros/indigo/setup.bash
```

Then, compile the code using:
```sh
mkdir build
cd build
cmake ..
make
```

Alternatively just call the build script which does all the above for you
```sh
./build.sh
```

### Optional Steps

##### To run all the unit/integration tests:
From the build directory:
```sh
make run_tests
```

##### To generate the documentation files using Doxygen:
From the build directory:
```sh
make docs
```
The documentation can be viewed by opening ```build/docs/html/index.html``` in your favorite browser.

##### To generate Eclipse project files :
From the top level directory:
```
python scripts/make_eclipse_project.py <new directory>
```
NOTE: The new directory should be OUTSIDE the top level directory (Eclipse does not like out-of-source project files in the source directory).
