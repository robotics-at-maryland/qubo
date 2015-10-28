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

include_directories(include ${catkin_INCLUDE_DIRS})

set(SRC_FILES
    src/vehicle_layer.cpp
    src/device.cpp
    src/example_device.cpp
    )
    
set(TEST_FILES
 
    )


##############################
# Add Executables ############
##############################

add_executable(vehicle_layer src/vehicle_layer.cpp) 
add_executable(example_device src/example_device.cpp)
target_link_libraries(example_device ${catkin_LIBRARIES})
target_link_libraries(vehicle_layer ${catkin_LIBRARIES})
  

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
