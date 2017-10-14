cmake_minimum_required(VERSION 2.8.3)
project(autonomy)

##############################
# Catkin Setup ###############
##############################

find_package(catkin

  REQUIRED COMPONENTS
  ram_msgs
  rospy

)


catkin_package(
  LIBRARIES
  ${PROJECT_NAME}
  CATKIN_DEPENDS
  cmake_modules
  rospy
  ram_msgs
  
)
