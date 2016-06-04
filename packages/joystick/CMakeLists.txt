cmake_minimum_required(VERSION 2.8.3)
project(joystick)

##############################
# Catkin Setup ###############
##############################


find_package(catkin 
    REQUIRED COMPONENTS
    roscpp
    cv_bridge #these two are used in the camera node
    image_transport
    ram_cpp_utils
    ram_drivers
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp
)


##############################
# File List ##################
##############################

include_directories(
  ${catkin_INCLUDE_DIRS}
  include
  )

set (JOY_SRC_FILES
  src/main.cpp
  src/joy_reader.cpp
)

set(TEST_FILES
  
    )

##############################
# Add Executables ############
##############################

add_executable(joy_node ${JOY_SRC_FILES}) 
target_link_libraries(joy_node ${catkin_LIBRARIES})

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
