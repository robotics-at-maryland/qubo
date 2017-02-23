cmake_minimum_required(VERSION 2.8.3)
project(vl_tortuga)

##############################
# Catkin Setup ###############
##############################

find_package(catkin
    REQUIRED COMPONENTS
    roscpp
    image_transport
    ram_cpp_utils
    ram_msgs
    tf2_ros
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp
)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

##############################
# File List ##################
##############################

include_directories(
  ${catkin_INCLUDE_DIRS}
  include

  drivers/include

  dvl/include
  imu/include

  sensor_board/include
)

set(DVL_SRC_FILES
  dvl/src/main.cpp
  dvl/src/dvl_tortuga.cpp

  drivers/src/dvlapi.c #OLD TEAMS DRIVER CODE
  )

set(IMU_SRC_FILES
  imu/src/main.cpp
  imu/src/imu_tortuga.cpp

  drivers/src/imuapi.c #OLD TEAMS DRIVER CODE
  )


set(SENSOR_BOARD_SRC_FILES
  sensor_board/src/temp_tortuga.cpp
  sensor_board/src/thruster_tortuga.cpp
  sensor_board/src/depth_tortuga.cpp
  sensor_board/src/power_tortuga.cpp
  #  sensor_board/src/sonar_client.cpp
  #  sensor_board/src/sonar_server.cpp
  sensor_board/src/sensor_board_tortuga.cpp
  sensor_board/src/main.cpp

  drivers/src/sensorapi.c #OLD TEAMS DRIVER CODE
  )


#### OLD TEAMS CODE ####
set(LCDSHOW_SRC_FILES
		drivers/src/lcdshow.c
		drivers/src/sensorapi.c

)
### END OF OLD TEAMS CODE ###

#SG: maybe one day..
set(TEST_FILES

  )

##############################
# Add Executables ############
##############################




#these get tortuga specific names to prevent conflicts with other vehicle layers
add_executable(tortuga_dvl_node_torguga ${DVL_SRC_FILES})
target_link_libraries(tortuga_dvl_node_torguga ${catkin_LIBRARIES})

add_executable(tortuga_imu_node ${IMU_SRC_FILES})
target_link_libraries(tortuga_imu_node ${catkin_LIBRARIES})

add_executable(lcdshow ${LCDSHOW_SRC_FILES})
target_link_libraries(lcdshow ${catkin_LIBRARIES})
add_executable(sensor_board_node ${SENSOR_BOARD_SRC_FILES})
target_link_libraries(sensor_board_node ${catkin_LIBRARIES})


##############################
# Unit Tests #################
##############################


##SG: maybe one day we'll have the time and inclination to make unit tests...

#catkin_add_gtest(${PROJECT_NAME}_tests ${TEST_FILES})

#set_target_properties(${PROJECT_NAME}_tests PROPERTIES LINKER_LANGUAGE CXX)

#target_link_libraries(${PROJECT_NAME}_tests
#    ${catkin_LIBRARIES}
#    ${PROJECT_NAME}
#)

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
