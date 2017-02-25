cmake_minimum_required(VERSION 2.8.3)
project(test)

##############################
# Catkin Setup ###############
##############################

find_package(catkin 
   # REQUIRED COMPONENTS
   # qubo_vehicle_layer
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS 
)


##############################
# File List ##################
##############################

include_directories(include ${catkin_INCLUDE_DIRS})

set(SRC_FILES
    src/helloworld.cpp
    )
    
set(TEST_FILES
 
    )

##############################
# Add Executables ############
##############################

add_executable(test src/helloworld.cpp)
target_link_libraries(test ${catkin_LIBRARIES})


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

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
