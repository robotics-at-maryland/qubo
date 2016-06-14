cmake_minimum_required(VERSION 2.8.3)
project(ram_drivers)


catkin_package(
    LIBRARIES ${PROJECT_NAME}
    INCLUDE_DIRS
    ahrs/include
    dvl/include
    tortuga/include
)


##############################
# File List ##################
##############################

include_directories(
  ahrs/include
  dvl/include
  tortuga/include
)
SET(CMAKE_C++_CREATE_SHARED_LIBRARY 1)

set(AHRS_SRC_FILES
        ahrs/src/AHRS.cpp
)

set(LCDSHOW_SRC_FILES
		tortuga/src/lcdshow.c
)

set(DVL_SRC_FILES
         dvl/src/DVL.cpp
)

set(TORTUGA_SRC_FILES
        tortuga/src/dvlapi.c
        tortuga/src/imuapi.c
        tortuga/src/lcdshow.c
        tortuga/src/sensorapi.c
)

set(AHRS_CONFIG_SRC_FILES
        ahrs/src/config.cpp
        ahrs/src/util.cpp
)

set(AHRS_BAUDRATE_SRC_FILES
        ahrs/src/baudrate.cpp
        ahrs/src/util.cpp
)





add_library(${PROJECT_NAME} ${DVL_SRC_FILES} ${AHRS_SRC_FILES} ${TORTUGA_SRC_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)


##############################
# Add Executables ############
##############################

add_executable(lcdshow ${LCDSHOW_SRC_FILES})
target_link_libraries(lcdshow ${PROJECT_NAME})

add_executable(ahrs_config ${AHRS_CONFIG_SRC_FILES})
target_link_libraries(ahrs_config ${PROJECT_NAME})

add_executable(ahrs_baudrate ${AHRS_BAUDRATE_SRC_FILES})
target_link_libraries(ahrs_baudrate ${PROJECT_NAME})


##############################
# Installable Targets ########
##############################


install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
