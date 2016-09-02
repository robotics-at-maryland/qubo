cmake_minimum_required(VERSION 2.8.3)
project(vision)

find_package(catkin 
  REQUIRED COMPONENTS
  roscpp
  roslib
  image_transport
  cv_bridge
)

find_package(OpenCV REQUIRED)

find_package(
  Boost
#  REQUIRED COMPONENTS algorithm
)

catkin_package(
   INCLUDE_DIRS
   LIBRARIES ${PROJECT_NAME}
   CATKIN_DEPENDS roscpp roslib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include/
)

add_executable(test src/test.cpp)
target_link_libraries(test ${catkin_LIBRARIES})

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

#target_link_libraries(vision_node ${catkin_LIBRARIES})
#target_link_libraries(vision_node ${OpenCV_LIBRARIES})
#

