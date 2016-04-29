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
    src/tortuga/old_code/sensorapi.c
    src/tortuga/old_code/imuapi.c
	#    src/tortuga/old_code/thrusterapi.c
	#src/tortuga/old_code/thrusterstop.c
    )

set(LCDSHOW_FILES
	src/tortuga/old_code/lcdshow.c
	)

set(BOARD_OPENER_FILES
    src/tortuga/new_code/board_opener.cpp)
    
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
# Add Executables ############
##############################

add_executable(board_opener ${BOARD_OPENER_FILES}) 
target_link_libraries(board_opener 
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}
)

add_executable(lcdshow ${LCDSHOW_FILES}) 
target_link_libraries(lcdshow
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}
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
