cmake_minimum_required(VERSION 2.8.3)
project(ram_msgs)

##############################
# Package Setup ##############
##############################

find_package(
    catkin REQUIRED COMPONENTS
    std_msgs
    sensor_msgs
    message_generation
    roscpp
    rospy

    #sg: action lib tutorial  told me to add these
     genmsg 
     actionlib_msgs 
     actionlib
)

##############################
# Messages #########
##############################

add_message_files(
    FILES
    Matrix2.msg
    Matrix3.msg
    Matrix4.msg
    Quaternion.msg
    Vector2.msg
    Vector3.msg
    Vector4.msg

    # New messages added in 2015
    DVL.msg

    PowerSource.msg
    Temp.msg
    Sim_Power_Source.msg
    Sim_Temperature.msg
    Sim_Led.msg
    Led.msg

    #Qubo specific messages
    DVL_qubo.msg

    #qubobus messages
    Status.msg
    )

##############################
# Services##########
##############################
add_service_files(
    FILES
    sonar_data.srv

    bool_bool.srv
)


##############################
#Actions##########
##############################

add_action_files(DIRECTORY action

  FILES 
  VisionExample.action
)

##############################
#Generation##########
##############################

generate_messages(
    DEPENDENCIES
    std_msgs
    actionlib_msgs
)

##############################
# Catkin Setup ###############
##############################

catkin_package(
    CATKIN_DEPENDS message_runtime std_msgs
)
