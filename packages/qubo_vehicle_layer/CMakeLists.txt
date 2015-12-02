cmake_minimum_required(VERSION 2.8.3)
project(qubo_vehicle_layer)

##############################
# Catkin Setup ###############
##############################


find_package(catkin 
    REQUIRED COMPONENTS
   # qubo_vehicle_layer
    roscpp
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
  imu/include)

set(DEPTH_SRC_FILES
  depth_sensor/src/main.cpp
  depth_sensor/src/depth_sensor_sim.cpp
  )

set(DVL_SRC_FILES
  dvl/src/main.cpp
  dvl/src/dvl_sim.cpp
  )

set(IMU_SRC_FILES
  imu/src/main.cpp
  imu/src/imu_sim.cpp
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
