cmake_minimum_required(VERSION 2.8.3)
project(ram_cpp_utils)

##############################
# Catkin Setup ###############
##############################

find_package(catkin
    REQUIRED COMPONENTS
    message_generation
    ram_msgs
    roscpp

)

catkin_package(
    INCLUDE_DIRS
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp #ram_msgs message_runtime
)

##############################
# File List ##################
##############################

include_directories(math/include include   ${catkin_INCLUDE_DIRS})

set(SRC_FILES
    math/src/Conversion.cpp
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

add_dependencies(${PROJECT_NAME}  ${catkin_EXPORTED_TARGETS})



##############################
# Add Executables ############
##############################


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
