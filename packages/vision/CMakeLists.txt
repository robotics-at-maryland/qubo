cmake_minimum_required(VERSION 2.8.3)
project(vision)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin 
  REQUIRED COMPONENTS
  roscpp
  roslib
  OpenCV
)

catkin_package(
   INCLUDE_DIRS #include
   LIBRARIES ${PROJECT_NAME}
   CATKIN_DEPENDS roscpp roslib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include/
)


add_executable(vision_node src/my_subscriber.cpp
  src/my_publisher.cpp
  src/main.cpp)

target_link_libraries(vision_node ${catkin_LIBRARIES})
target_link_libraries(vision_node ${OpenCV_LIBRARIES})

