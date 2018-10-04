ke_minimum_required(VERSION 2.8.3)
   2 project(beginner_tutorials)
   3 
   4 ## Find catkin and any catkin packages
   5 find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)
   6 
   7 ## Declare ROS messages and services
   8 add_message_files(FILES Num.msg)
   9 add_service_files(FILES AddTwoInts.srv)
  10 
  11 ## Generate added messages and services
  12 generate_messages(DEPENDENCIES std_msgs)
  13 
  14 ## Declare a catkin package
  15 catkin_package()
  16 
  17 ## Build talker and listener
  18 include_directories(include ${catkin_INCLUDE_DIRS})
  19 
  20 add_executable(talker src/talker.cpp)
  21 target_link_libraries(talker ${catkin_LIBRARIES})
  22 add_dependencies(talker beginner_tutorials_generate_messages_cpp)
  23 
  24 add_executable(listener src/listener.cpp)
  25 target_link_libraries(listener ${catkin_LIBRARIES})
  26 add_dependencies(listener beginner_tutorials_generate_messages_cpp)
