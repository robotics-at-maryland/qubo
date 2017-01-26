cmake_minimum_required(VERSION 2.8.3) 
project(controls)

##############################
# Catkin Setup ###############
##############################

find_package(catkin REQUIRED COMPONENTS
    roscpp
    cv_bridge #these two are used in the camera node
    image_transport
    ram_cpp_utils
    ram_msgs
    tf2_ros
)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp
)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

##############################
# File List ##################
##############################

include_directories(
  ${catkin_INCLUDE_DIRS}
  include/
)

set(PID_CONTROLLER_SRC_FILES
  src/pid_controller.cpp
  src/pid_controller_node.cpp
)

##############################
# Add Executables ############
##############################

add_executable(pid_controller_node ${PID_CONTROLLER_SRC_FILES})
target_link_libraries(pid_controller_node ${catkin_LIBRARIES})

##############################
# Installable Targets ########
##############################

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
