#ifndef OPEN_LOOP_TELEOP_H
#define OPEN_LOOP_TELEOP_H

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Float64MultiArray.h>

#define MAX_THRUSTER_INPUT 255

class OpenLoopTeleop {
    public:
        OpenLoopTeleop(int);
        ~OpenLoopTeleop();		
        void update();
    private:
        ros::NodeHandle nh;
        std_msgs::Float64MultiArray thruster_input;
        ros::Subscriber joy_sub;
        ros::Publisher thruster_pub;
        void joyInputCallback(const sensor_msgs::Joy::ConstPtr &);
};

#endif
