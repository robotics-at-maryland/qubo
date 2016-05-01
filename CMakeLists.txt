cmake_minimum_required(VERSION 2.8.3)
project("qubo")

#Initialize various variables with path locations.
set(RAM_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RAM_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_MODULE_PATH ${RAM_TOP_DIR}/cmake)

#Set default values for CATKIN_DEVEL_PREFIX and CMAKE_INSTALL_PREFIX
if(NOT DEFINED CATKIN_DEVEL_PREFIX)
    set(CATKIN_DEVEL_PREFIX ${RAM_TOP_DIR}/devel)
endif()
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${RAM_TOP_DIR}/install)
endif()

#Quick fix for building without sourcing devel/setup.bash
include_directories(
    $ENV{ROS_ROOT}/../../include
    ${CATKIN_DEVEL_PREFIX}/include
)

#Enable C++11 or C++0x support.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(SEND_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#Find Doxygen and create a "docs" target for generating the documentation.
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_MODULE_PATH}/templates/Doxyfile.in ${RAM_BIN_DIR}/Doxyfile @ONLY)
    add_custom_target(
        docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${RAM_BIN_DIR}/Doxyfile
        WORKING_DIRECTORY ${RAM_TOP_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

#Initialize the Catkin workspace if it is not set up.
if(EXISTS ${RAM_TOP_DIR}/packages/CMakeLists.txt)
    message(STATUS "Catkin workspace is already set up.")
else()
    message(STATUS "Catkin workspace is NOT set up.  Initializing...")
    execute_process(
        COMMAND catkin_init_workspace
        WORKING_DIRECTORY ${RAM_TOP_DIR}/packages
    )
endif()

#Configure a .pydevproject file if we are also generating Eclipse project files.
if(CMAKE_ECLIPSE_VERSION)
    message(STATUS "Configuring Eclipse .pydevproject file...")
    configure_file(${CMAKE_MODULE_PATH}/templates/.pydevproject.in ${RAM_BIN_DIR}/.pydevproject @ONLY)
endif()

add_subdirectory(drivers)
add_subdirectory(packages)
