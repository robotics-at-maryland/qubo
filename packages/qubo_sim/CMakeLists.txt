cmake_minimum_required(VERSION 2.8.3)
project(qubo_sim)

##############################
# Catkin Setup ###############
##############################

find_package(catkin 
   REQUIRED COMPONENTS
   # qubo_vehicle_layer
   roscpp
   ram_msgs
   message_generation
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp message_runtime
)


##############################
# File List ##################
##############################

include_directories(include ${catkin_INCLUDE_DIRS})

set(SRC_FILES
    src/battery_sensor_sim.cpp
    src/temperature_sensor_sim.cpp
    src/led_status_sensor.cpp
    )
    
set(TEST_FILES
 
    )

##############################
# Add Executables ############
##############################

add_executable(battery_sensor_sim src/battery_sensor_sim.cpp)
target_link_libraries(battery_sensor_sim ${catkin_LIBRARIES})

add_executable(temperature_sensor_sim src/temperature_sensor_sim.cpp)
target_link_libraries(temperature_sensor_sim ${catkin_LIBRARIES})

add_executable(led_status_sensor src/led_status_sensor.cpp)
target_link_libraries(led_status_sensor ${catkin_LIBRARIES})


##############################
# Unit Tests #################
##############################

#catkin_add_gtest(${PROJECT_NAME}_tests ${TEST_FILES})

#set_target_properties(${PROJECT_NAME}_tests PROPERTIES LINKER_LANGUAGE CXX)

#target_link_libraries(${PROJECT_NAME}_tests
#    ${catkin_LIBRARIES}
#    ${PROJECT_NAME}
#)

##############################
# Installable Targets ########
##############################


#SG: Not too sure what stephen had these doing, but they break the build so they got to go
#install(TARGETS ${PROJECT_NAME}
#  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
