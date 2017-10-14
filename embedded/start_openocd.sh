#!/bin/bash
#
# Usage: start_openocd.sh [board_config_file]
#
# Starts an instance of OpenOCD and uses the specified configuration
# file. If no configuration file is specified,
# "/usr/local/share/openocd/scripts/board/ek-tm4c123gxl.cfg"
# will be used, i.e. the configuration file for the Tiva TM4C123GXL
# Launchpad at its default install location.
#
# When OpenOCD starts successfully, it can be accessed via Telnet
# at the TCP port 4444. 
#
# GDB debugger connects to OpenOCD via port 3333.
# When an instance of GDB is run on an ELF file, apply the
# following command to connect to OpenOCD:
#      target extended-remote localhost:3333 
#


# It is highly recommended to install the most recent version of OpenOCD,
# available at the SourceForge Git repository. It should be installed
# as described below:
# 
#   git clone http://git.code.sf.net/p/openocd/code
#   cd code
#   ./bootstrap
#   ./configure --enable-maintainer-mode --enable-ti-icdi --enable-ftdi --enable-jlink
#   make
#   make install
#
# Note: "make install" must be run with superuser privileges and will 
# install OpenOCD to the default install location. If you wish to install
# it elsewhere, provide additional arguments to 'configure'.
# Run "./configure --help" for more details.


# Default CFG_FILE:
CFG_FILE=/usr/local/share/openocd/scripts/board/ek-tm4c123gxl.cfg

# Assign the first argument to CFG_FILE if at least one CLI argument is provided:
if [ $# -ge 1 ]; then
    CFG_FILE=$1;
fi

# And finally start OpenOCD:

openocd --file $CFG_FILE #--enable-ioutil
