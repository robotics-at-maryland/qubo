cmake_minimum_required(VERSION 2.8.3)
project(vl_qubo)

find_package(catkin
  REQUIRED COMPONENTS
  roscpp
  ram_msgs
  )


catkin_package(
  LIBRARIES ${PROJECT_NAME}
  INCLUDE_DIRS
  CATKIN_DEPENDS roscpp
  )


##############################
# File List ##################
##############################

include_directories(
  ${catkin_INCLUDE_DIRS}
  drivers/ahrs/include
  drivers/dvl/include
  drivers/qscu/include
  drivers/qubobus/include

  include

  )

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

SET(CMAKE_C++_CREATE_SHARED_LIBRARY 1)

#SG: currently unused, eventually we'll add a wrapper with a main and build them together.
set(AHRS_SRC_FILES

  src/ahrs_main.cpp
  src/ahrs_node.cpp

  drivers/ahrs/src/AHRS.cpp
  )

#SG: ditto above
set(DVL_SRC_FILES

  src/dvl_main.cpp
  src/dvl_node.cpp

  drivers/dvl/src/DVL.cpp
  )

set(AHRS_CONFIG_SRC_FILES
  drivers/ahrs/src/config.cpp
  drivers/ahrs/src/util.cpp

  drivers/ahrs/src/AHRS.cpp # SG: maybe putting this here is not the best way to do this, but it will work for now
  )

set(AHRS_BAUDRATE_SRC_FILES
  drivers/ahrs/src/baudrate.cpp
  drivers/ahrs/src/util.cpp


  drivers/ahrs/src/AHRS.cpp #SG: ditto above
  )


set(QSCU_SRC_FILES
  drivers/qscu/src/QSCU.cpp
  drivers/qscu/src/main.cpp
  )

set(QSCU_CONTROL_NODE_SRC_FILES
  drivers/qscu/src/qscu_node.cpp
  drivers/qscu/src/qscu_node_main.cpp
  drivers/qscu/src/QSCU.cpp
  )

##sg: This does the same as set but it allows us to match everything in the
#source directory
file(GLOB QUBOBUS_LIB_FILES
  drivers/qubobus/src/*.c
  )

##Qubobus Test Files

set(IO_TEST_FILES
  drivers/qubobus/test/test_io.c
  )

set(DEF_TEST_FILES
  drivers/qubobus/test/test_defs.c
  )

##############################
# Add Executables ############
##############################


add_executable(ahrs_config ${AHRS_CONFIG_SRC_FILES})
#target_link_libraries(ahrs_config ${PROJECT_NAME})

add_executable(ahrs_baudrate ${AHRS_BAUDRATE_SRC_FILES})
#target_link_libraries(ahrs_baudrate ${PROJECT_NAME})

add_executable(qubo_ahrs_node ${AHRS_SRC_FILES})
target_link_libraries(qubo_ahrs_node ${catkin_LIBRARIES})

add_executable(qubo_dvl_node ${DVL_SRC_FILES})
target_link_libraries(qubo_dvl_node ${catkin_LIBRARIES})
add_dependencies(qubo_dvl_node ram_msgs_generate_messages_cpp)

add_library(qubobus ${QUBOBUS_LIB_FILES})
set_target_properties(qubobus PROPERTIES LINKER_LANGUAGE C)

add_executable(test_io ${IO_TEST_FILES})
target_link_libraries(test_io qubobus)

add_executable(test_defs ${DEF_TEST_FILES})
target_link_libraries(test_defs qubobus)

#will probably change this back to a library at some point
add_executable(qscu ${QSCU_SRC_FILES})
target_link_libraries(qscu qubobus)

add_executable(qscu_node ${QSCU_CONTROL_NODE_SRC_FILES})
target_link_libraries(qscu_node qubobus ${catkin_LIBRARIES})
add_dependencies(qscu_node ram_msgs_generate_messages_cpp)

# catkin_install_python(PROGRAMS src/arduino_node.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


##############################
# Installable Targets ########
##############################



#sg: if any other package wants to use the libraries built here they need
#to uncomment this next part

# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
#   )

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
