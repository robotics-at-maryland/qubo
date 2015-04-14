# Project QUBO

## Setup Instructions

Project QUBO currently only supports Ubuntu 14.04.  Any other platforms may not (and probably will not) work correctly.

### Compilation

First of all, install all the dependencies in the project by running the handy installation script:
```sh
bash scripts/install_dependencies.bash
```

Then, compile the code using:
```sh
mkdir build
cd build
cmake ..
make
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
