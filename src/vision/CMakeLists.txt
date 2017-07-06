cmake_minimum_required(VERSION 2.8.3)
project(vision)


#Find stuff we need

find_package(catkin
  REQUIRED COMPONENTS
  roscpp
  roslib

  ram_msgs
  genmsg

  rospy
  std_msgs
  message_generation

  actionlib_msgs
  actionlib

  image_transport

  cv_bridge


)

#we found putting this later fixed the problem of some people's machines forgetting where OpenCV lives
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

#find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui opencv_gpu video)

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
  ${VISP_INCLUDE_DIRS}
  include/
  )

set(VISION_SRC_FILES
  src/main.cpp
  src/vision_node.cpp
  
  src/actions/buoy_action.cpp
  src/actions/gate_action.cpp
  
  src/tuners/buoy_tuner.cpp
  )


set(SIM_VISION_SRC_FILES
  src/sim_main.cpp
  src/sim_vision_node.cpp
  
  src/actions/buoy_action.cpp
  src/actions/gate_action.cpp
  )


set(GPU_VISION_SRC_FILES
  src/gpu_main.cpp
  src/gpu_vision_node.cpp
  )

##Tuners

set(CANNY_TUNER_FILES
  src/tuners/canny_tuner.cpp
  )


set(BLOB_TUNER_FILES
  src/tuners/blob_tuner.cpp
  )


set(GATE_TUNER_FILES
  src/tuners/gate_tuner.cpp
  )



#Prototype algorithms
set(GATE_FILES
  src/prototypes/gate_detector.cpp
  )


set(SEGMENT_FILES
  src/prototypes/segmentation.cpp
  )

set(EDGE_FILES
  src/prototypes/edge_detection.cpp
  )

set(BLOB_FILES
  src/prototypes/blob.cpp
  )

set(EM_FILES
 src/prototypes/em.cpp
 )

set(GFIT_FILES
  src/prototypes/g_fit.cpp
 )

set(MBT_FILES
  src/prototypes/mbt.cpp
  )

#Cascades

set(CASCADE_TEST
  src/cascades/CascadeTest.cpp
  )

##############################
# Add Executables ############
##############################



add_executable(vision_node ${VISION_SRC_FILES})
target_link_libraries(vision_node ${catkin_LIBRARIES})
target_link_libraries(vision_node ${OpenCV_LIBRARIES})
add_dependencies(vision_node ram_msgs_gencpp) #We need these built before we try to build this package

#still needs to be updated to 3.0
add_executable(gpu_vision_node ${GPU_VISION_SRC_FILES})
target_link_libraries(gpu_vision_node ${catkin_LIBRARIES})
target_link_libraries(gpu_vision_node ${OpenCV_LIBRARIES})
add_dependencies(gpu_vision_node ram_msgs_gencpp) #We need these built before we try to build this package



add_executable(sim_vision_node ${SIM_VISION_SRC_FILES})
target_link_libraries(sim_vision_node ${catkin_LIBRARIES})
target_link_libraries(sim_vision_node ${OpenCV_LIBRARIES})
add_dependencies(sim_vision_node ram_msgs_gencpp) #We need these built before we try to build this package



##below are all basically prototypes
add_executable(gate_detector ${GATE_FILES})
target_link_libraries(gate_detector ${OpenCV_LIBRARIES})


add_executable(canny_tuner ${CANNY_TUNER_FILES})
target_link_libraries(canny_tuner ${OpenCV_LIBRARIES})

add_executable(blob_tuner ${BLOB_TUNER_FILES})
target_link_libraries(blob_tuner ${OpenCV_LIBRARIES})

add_executable(gate_tuner ${GATE_TUNER_FILES})
target_link_libraries(gate_tuner ${OpenCV_LIBRARIES})

add_executable(segment_test ${SEGMENT_FILES})
target_link_libraries(segment_test ${OpenCV_LIBRARIES})

add_executable(canny_test ${EDGE_FILES})
target_link_libraries(canny_test ${OpenCV_LIBRARIES})

add_executable(blob_test ${BLOB_FILES})
target_link_libraries(blob_test ${OpenCV_LIBRARIES})

add_executable(em_test ${EM_FILES})
target_link_libraries(em_test ${OpenCV_LIBRARIES})

add_executable(gfit_test ${GFIT_FILES})
target_link_libraries(gfit_test ${OpenCV_LIBRARIES})

# add_executable(mbt_test ${MBT_FILES})
# target_link_libraries(mbt_test ${VISP_LIBRARIES})

add_executable(cascade_test ${CASCADE_TEST})
target_link_libraries(cascade_test ${OpenCV_LIBS})



##############################
# Installable Targets ########
##############################

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
