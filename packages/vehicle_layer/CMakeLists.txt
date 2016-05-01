cmake_minimum_required(VERSION 2.8.3)
project(qubo_vehicle_layer)

##############################
# Catkin Setup ###############
##############################


find_package(catkin 
    REQUIRED COMPONENTS
   # qubo_vehicle_layer
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
  depth_sensor/include
  dvl/include
  imu/include
  power_source/include
  led/include
  temperature/include
  camera/include
  thrusters/include
  sensor_board/include
  )

set(DEPTH_SRC_FILES
  depth_sensor/src/main.cpp
  depth_sensor/src/depth_sensor_sim.cpp
  depth_sensor/src/depth_sensor_tortuga.cpp
  )

set(DVL_SRC_FILES
  dvl/src/main.cpp
  dvl/src/dvl_sim.cpp
  dvl/src/dvl_tortuga.cpp
  )

set(IMU_SRC_FILES
  imu/src/main.cpp
  imu/src/imu_sim.cpp
  imu/src/imu_tortuga.cpp
  )

set(POWER_SRC_FILES
  power_source/src/main.cpp
  power_source/src/power_sensor_sim.cpp
  power_source/src/power_sensor_tortuga.cpp
  )

set(LED_SRC_FILES
  led/src/main.cpp
  led/src/led_sim.cpp
  )

set(TEMP_SRC_FILES
  temperature/src/main.cpp
  temperature/src/temp_sensor_sim.cpp
  )

set(CAMERA_SRC_FILES
  camera/src/main.cpp
  camera/src/camera_sim.cpp
  )

set(THRUSTER_SRC_FILES
  thrusters/src/main.cpp
  thrusters/src/thruster_sim.cpp
  thrusters/src/thruster_tortuga.cpp
  )

set(SENSOR_BOARD_SRC_FILES
  sensor_board/src/main.cpp
  sensor_board/src/sensor_board_tortuga.cpp
  )

set(TEST_FILES
  
    )

##############################
# Add Executables ############
##############################


add_executable(depth_node ${DEPTH_SRC_FILES}) 
target_link_libraries(depth_node ${catkin_LIBRARIES})

add_executable(dvl_node ${DVL_SRC_FILES})
target_link_libraries(dvl_node ${catkin_LIBRARIES})

add_executable(imu_node ${IMU_SRC_FILES})
target_link_libraries(imu_node ${catkin_LIBRARIES})
  
add_executable(power_node ${POWER_SRC_FILES})
target_link_libraries(power_node ${catkin_LIBRARIES})

add_executable(led_node ${LED_SRC_FILES})
target_link_libraries(led_node ${catkin_LIBRARIES})

add_executable(temp_node ${TEMP_SRC_FILES})
target_link_libraries(temp_node ${catkin_LIBRARIES})

add_executable(camera_node ${CAMERA_SRC_FILES})
target_link_libraries(camera_node ${catkin_LIBRARIES})

add_executable(thruster_node ${THRUSTER_SRC_FILES})
target_link_libraries(thruster_node ${catkin_LIBRARIES})

add_executable(sensor_board_node ${SENSOR_BOARD_SRC_FILES})
target_link_libraries(sensor_board_node ${catkin_LIBRARIES})


##############################
# Unit Tests #################
##############################


##SG: PLEASE ADD THESE WHEN YOU MAKE SOMETHING FOR REAL

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
