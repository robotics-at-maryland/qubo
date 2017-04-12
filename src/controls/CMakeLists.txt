cmake_minimum_required(VERSION 2.8.3)
project(controls)

##############################
# Catkin Setup ###############
##############################

find_package(catkin REQUIRED COMPONENTS
  ram_cpp_utils
  ram_msgs
  roscpp
  tf2_ros

)

find_package(Eigen3)

catkin_package(
  LIBRARIES
  ${PROJECT_NAME}
  CATKIN_DEPENDS
  cmake_modules
  roscpp
  ram_cpp_utils
  ram_msgs
  DEPENDS EIGEN3
)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

##############################
# File List ##################
##############################

include_directories(
  ${catkin_INCLUDE_DIRS}
  include/
)


##############################
# Add Executables ############
##############################

add_executable(pid_controller_node
  src/pid_controller.cpp
  src/pid_controller_node.cpp
  )

target_link_libraries(pid_controller_node
  ${catkin_LIBRARIES}
)

##############################
# Installable Targets ########
##############################



#SG: I'll be honest, I don't entirely know what this commented out code does, but it breaks when I leave it in
#install(TARGETS ${PROJECT_NAME}
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)



install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
