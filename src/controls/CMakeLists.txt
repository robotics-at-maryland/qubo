cmake_minimum_required(VERSION 2.8.3)
project(controls)

##############################
# Catkin Setup ###############
##############################

find_package(catkin
  REQUIRED COMPONENTS
  ram_msgs
  roscpp
  tf2_ros
  dynamic_reconfigure
)

find_package(Eigen3)

#find_package(catkin REQUIRED dynamic_reconfigure)
generate_dynamic_reconfigure_options(
  cfg/pid.cfg
)


catkin_package(
  LIBRARIES
  ${PROJECT_NAME}
  CATKIN_DEPENDS
  cmake_modules
  roscpp
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


set(PID_CONTROLLER_FILES
  src/pid_controller.cpp
  src/pid_main.cpp
  )

set(ROT_CONTROLLER_FILES
  src/pid_controller.cpp
  src/rot_controller.cpp
  src/rot_main.cpp
  )

##############################
# Add Executables ############
##############################

add_executable(pid_controller ${PID_CONTROLLER_FILES})
target_link_libraries(pid_controller ${catkin_LIBRARIES})
add_dependencies(pid_controller ${PROJECT_NAME}_gencfg)

add_executable(rot_controller ${ROT_CONTROLLER_FILES})
target_link_libraries(rot_controller ${catkin_LIBRARIES})
add_dependencies(rot_controller ${PROJECT_NAME}_gencfg)


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
