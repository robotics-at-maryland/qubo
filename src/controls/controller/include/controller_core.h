#ifndef CONTROLLER_CORE_H
#define CONTROLLER_CORE_H

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "std_msgs/Float64MultiArray.h"
#include "nav_msgs/Odometry.h"

class controlNode {
    
    public:
    //! Constructor.
    controlNode(std::shared_ptr<ros::NodeHandle>  , int);
		
    void update();
    //! Destructor.
    ~controlNode();
    
    private:
    
    //! Callback function for subscriber.
    void messageCallback(const std_msgs::Float64MultiArray::ConstPtr &msg);
    ros::Subscriber joystick_sub;
    ros::Publisher next_state_pub;
    ros::Time current_time;
    float dt, th, dx, dy, dz, vth, vx, vy, vz;
    geometry_msgs::Quaternion next_state_orient;
};

#endif // MOVE_CORE_H

