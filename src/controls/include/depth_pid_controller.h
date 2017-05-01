#ifndef DEPTH_CONTROLLER_H
#define DEPTH_CONTROLLER_H

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int64MultiArray.h>


class DepthController {
    public:
    DepthController(ros::NodeHandle *nh);
    ~DepthController();

    protected:
    void update();

    ros::Subscriber _sensor_sub;
    void depthSensorCallback(const std_msgs::Float64::ConstPtr& msg);
    double _current_depth;

	ros::Subscriber _command_pub; 
    void depthCommandhCallback(const std_msgs::Float64::ConstPtr& msg);
    double _depth_command;
    
#endif //PID_CONTROLLER_H
