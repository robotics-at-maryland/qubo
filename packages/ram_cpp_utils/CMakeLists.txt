cmake_minimum_required(VERSION 2.8.3)
project(ram_cpp_utils)

##############################
# Catkin Setup ###############
##############################

find_package(catkin 
    REQUIRED COMPONENTS
    ram_msgs
    roscpp
    cv_bridge
    image_transport
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
    src/math/Conversion.cpp
    src/tortuga/sensorapi.c
    )
    
set(TEST_FILES
    test/ConversionTests.cpp
    test/TestMain.cpp
    )

##############################
# Library ####################
##############################

add_library(${PROJECT_NAME} ${SRC_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

##############################
# Unit Tests #################
##############################

catkin_add_gtest(${PROJECT_NAME}_tests ${TEST_FILES})

set_target_properties(${PROJECT_NAME}_tests PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(${PROJECT_NAME}_tests
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}
)

##############################
# Installable Targets ########
##############################

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
