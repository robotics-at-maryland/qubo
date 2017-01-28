cmake_minimum_required(VERSION 2.8.3)
project(vision)

#Find stuff we need
find_package(catkin
  REQUIRED COMPONENTS
  roscpp
  roslib
  image_transport
  cv_bridge
  ram_msgs
  genmsg

  OpenCV

  rospy
  std_msgs
  message_generation

  actionlib_msgs
  actionlib
)

#sg: these used to be here but I have reason to believe they do not do anything, so I'm getting rid of them
#find_package(OpenCV REQUIRED)

#find_package(
#  Boost
#  REQUIRED COMPONENTS algorithm
#)

#generate messages used by the server and client from vison.action
add_action_files(DIRECTORY ../ram_msgs/action FILES VisionExample.action)
#the files used by services
add_service_files(DIRECTORY ../ram_msgs/srv FILES bool_bool.srv)
generate_messages(DEPENDENCIES actionlib_msgs std_msgs)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

#Stuff other packages that depend on us need
catkin_package(
   INCLUDE_DIRS
   LIBRARIES ${PROJECT_NAME}
   CATKIN_DEPENDS roscpp roslib actionlib_msgs
)


##############################
# File List ##################
##############################

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include/
)

set(VISION_SRC_FILES
  src/main.cpp
  src/vision_node.cpp
  )

set(GPU_VISION_SRC_FILES
  src/gpu_main.cpp
  src/gpu_vision_node.cpp
  )

set(CANNY_TUNER_FILES
  src/canny_tuner.cpp
  )


set(SEGMENT_FILES
  src/prototypes/segmentation.cpp
  )

set(EDGE_FILES
  src/prototypes/edge_detection.cpp
  )

set(MOG_FILES
  src/prototypes/mog.cpp
  )

set(EM_FILES
  src/prototypes/em.cpp
  )

##############################
# Add Executables ############
##############################

add_executable(gpu_vision_node ${GPU_VISION_SRC_FILES})
target_link_libraries(gpu_vision_node ${catkin_LIBRARIES})
target_link_libraries(gpu_vision_node ${OpenCV_LIBRARIES})

add_executable(vision_node ${VISION_SRC_FILES})
target_link_libraries(vision_node ${catkin_LIBRARIES})
target_link_libraries(vision_node ${OpenCV_LIBRARIES})
add_dependencies(vision_node ram_msgs_gencpp) #We need these built before we try to build this package


##below are all basically prototypes

add_executable(canny_tuner ${CANNY_TUNER_FILES})
target_link_libraries(canny_tuner ${OpenCV_LIBRARIES})

add_executable(segment_test ${SEGMENT_FILES})
target_link_libraries(segment_test ${OpenCV_LIBRARIES})

add_executable(canny_test ${EDGE_FILES})
target_link_libraries(canny_test ${OpenCV_LIBRARIES})

add_executable(mog_test ${MOG_FILES})
target_link_libraries(mog_test ${OpenCV_LIBRARIES})

add_executable(em_test ${EM_FILES})
target_link_libraries(em_test ${OpenCV_LIBRARIES})


##############################
# Installable Targets ########
##############################

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
