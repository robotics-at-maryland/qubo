cmake_minimum_required(VERSION 2.8.3)
project(ram_msgs)

##############################
# Package Setup ##############
##############################

find_package(
    catkin REQUIRED COMPONENTS 
    message_generation
)

##############################
# Message Generation #########
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

    Sim_Power_Source.msg
)

generate_messages()

##############################
# Catkin Setup ###############
##############################

catkin_package(
    CATKIN_DEPENDS message_runtime
)
