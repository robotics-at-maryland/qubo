cmake_minimum_required(VERSION 2.8.3)
project(vl_qubo_gazebo)

## Add support for C++11, supported in ROS Kinetic and newer
# add_definitions(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin
  REQUIRED COMPONENTS
  roscpp
  ram_msgs
  uuv_gazebo_ros_plugins_msgs
  )


catkin_package(
  LIBRARIES ${PROJECT_NAME}
  INCLUDE_DIRS
  CATKIN_DEPENDS roscpp
  )


##############################
# File List ##################
##############################

include_directories(
  ${catkin_INCLUDE_DIRS}
  drivers/ahrs/include
  drivers/dvl/include

  ahrs/include
  dvl/include
  include

  )

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")


set(G_HARDWARE_NODE_SRC_FILES
  src/gazebo_main.cpp
  src/gazebo_hardware_node.cpp
  )


##############################
# Add Executables ############
##############################

add_executable(qubo_gazebo_hardware_node ${G_HARDWARE_NODE_SRC_FILES})
target_link_libraries(qubo_gazebo_hardware_node ${catkin_LIBRARIES})

##############################
# Installable Targets ########
##############################


#install(TARGETS ${PROJECT_NAME}
#       ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#       LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#      RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
#)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
