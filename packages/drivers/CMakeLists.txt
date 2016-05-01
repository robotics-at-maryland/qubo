cmake_minimum_required(VERSION 2.8.3)
project(ram_drivers)


catkin_package(
    LIBRARIES ${PROJECT_NAME}
    INCLUDE_DIRS ahrs/include
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

set(AHRS_SRC_FILES
        ahrs/src/AHRS.cpp
  
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
set(AHRS_TEST_SRC_FILES
        ahrs/src/test.cpp
        ahrs/src/util.cpp
)


set(DVL_CONFIG_SRC_FILES
        dvl/src/config.cpp
        dvl/src/util.cpp
)
set(DVL_BAUDRATE_SRC_FILES
        dvl/src/baudrate.cpp
        dvl/src/util.cpp
)
set(DVL_TEST_SRC_FILES
        dvl/src/test.cpp
)


add_library(${PROJECT_NAME} ${DVL_SRC_FILES} ${AHRS_SRC_FILES} ${TORTUGA_SRC_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE C++) 

add_executable(ahrs_config ${AHRS_CONFIG_SRC_FILES})
target_link_libraries(ahrs_config ${PROJECT_NAME})

add_executable(ahrs_baudrate ${AHRS_BAUDRATE_SRC_FILES})
target_link_libraries(ahrs_baudrate ${PROJECT_NAME})

add_executable(ahrs_test ${AHRS_TEST_SRC_FILES})
target_link_libraries(ahrs_test ${PROJECT_NAME})



add_executable(dvl_config ${DVL_CONFIG_SRC_FILES})
target_link_libraries(dvl_config  ${PROJECT_NAME})

add_executable(dvl_baudrate ${DVL_BAUDRATE_SRC_FILES})
target_link_libraries(dvl_baudrate  ${PROJECT_NAME})


#SG: currently does not build but I want to move on to other things. 
#add_executable(dvl_test ${DVL_TEST_SRC_FILES})
#target_link_libraries(dvl_test  ${PROJECT_NAME}) 

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
