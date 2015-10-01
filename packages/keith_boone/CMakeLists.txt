cmake_minimum_required(VERSION 2.8.3)
project(keith_boone)

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
    src/sean_is_a_loser.cpp
    )
    
set(TEST_FILES
 
    )

##############################
# Add Executables ############
##############################

add_executable(keith_boone src/sean_is_a_loser.cpp)
target_link_libraries(keith_boone ${catkin_LIBRARIES})


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
