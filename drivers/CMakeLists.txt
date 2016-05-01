cmake_minimum_required(VERSION 2.8.3)
project(ram_drivers)

##############################
# File List ##################
##############################

include_directories(
  ahrs/include
  dvl/include
  tortuga/include
)

set(AHRS_SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/ahrs/src/AHRS.cpp
        # ahrs/src/impl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ahrs/src/util.cpp
)

set(DVL_SRC_FILES    
         ${CMAKE_CURRENT_SOURCE_DIR}/dvl/src/DVL.cpp
       # dvl/src/impl.cpp
       # dvl/src/statics.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/dvl/src/util.cpp
)

set(TORTUGA_SRC_FILES
        tortuga/src/dvlapi.c
        tortuga/src/imuapi.c
        tortuga/src/lcdshow.c
        tortuga/src/sensorapi.c
)            
           



set(AHRS_CONFIG_SRC_FILES
        ahrs/src/config.cpp
)
set(AHRS_BAUDRATE_SRC_FILES
        ahrs/src/baudrate.cpp
)
set(AHRS_TEST_SRC_FILES
        ahrs/src/test.cpp
)


set(DVL_CONFIG_SRC_FILES
        ahrs/src/config.cpp
)
set(DVL_BAUDRATE_SRC_FILES
        ahrs/src/baudrate.cpp
)
set(DVL_TEST_SRC_FILES
        ahrs/src/test.cpp
)


add_library(dvl_driver ${DVL_SRC_FILES})
#set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

add_library(ahrs_driver ${AHRS_SRC_FILES})
#set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
     
add_library(tortuga_drivers ${TORTUGA_SRC_FILES})
#set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)


add_executable(ahrs_config ${AHRS_CONFIG_SRC_FILES})
target_link_libraries(ahrs_config ahrs_driver)

add_executable(ahrs_baudrate ${AHRS_BAUDRATE_SRC_FILES})
target_link_libraries(ahrs_baudrate ahrs_driver)

add_executable(ahrs_test ${AHRS_TEST_SRC_FILES})
target_link_libraries(ahrs_test ahrs_driver)



add_executable(dvl_config ${DVL_CONFIG_SRC_FILES})
target_link_libraries(dvl_config dvl_driver)

add_executable(dvl_baudrate ${DVL_BAUDRATE_SRC_FILES})
target_link_libraries(dvl_baudrate dvl_driver)

add_executable(dvl_test ${DVL_TEST_SRC_FILES})
target_link_libraries(dvl_test dvl_driver) 



install(TARGETS ahrs_driver dvl_driver tortuga_drivers
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
