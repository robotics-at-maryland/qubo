cmake_minimum_required(VERSION 2.8.3)
project(controls)

##############################
# Catkin Setup ###############
##############################

find_package(catkin
    REQUIRED COMPONENTS
    roscpp
    #cv_bridge #these two are used in the camera node
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
  controller/include
  localization/include
  angular_controller/include
  translational_controller/include

)

set(TRANSLATION_SRC_FILES
  translational_controller/src/linear_controller.cpp
  translational_controller/src/main.cpp
  )

set(ANGULAR_SRC_FILES
  angular_controller/src/rotational_controller.cpp
  angular_controller/src/main.cpp
)

set(CONTROLLER_NODE_SRC_FILES
  controller/src/controller_node.cpp
  controller/src/main.cpp
)

#### OLD TEAMS CODE ####

set(LCDSHOW_SRC_FILES
		tortuga/src/lcdshow.c
)


### END OF OLD TEAMS CODE ###

#SG: maybe one day..
set(TEST_FILES

    )

##############################
# Add Executables ############
##############################


add_executable(angular_control_node ${ANGULAR_SRC_FILES})
target_link_libraries(angular_control_node ${catkin_LIBRARIES})

add_executable(translational_control_node ${TRANSLATION_SRC_FILES})
target_link_libraries(translational_control_node ${catkin_LIBRARIES})

add_executable(controller_node ${CONTROLLER_NODE_SRC_FILES})
target_link_libraries(controller_node ${catkin_LIBRARIES})

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
