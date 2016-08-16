#ifndef ROTATIONAL_CONTROLLER_H
#define ROTATIONAL_CONTROLLER_H

// ROS includes.
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "ram_node.h"
#include <nav_msgs/Odometry.h>
#include "std_msgs/Int64MultiArray.h"
//#include "tortuga/sensorapi.h"


// Custom message includes. Auto-generated from msg/ directory.
#include <sensor_msgs/Joy.h>

#define MAX_THRUSTER_SPEED 255

class RotationalController : public RamNode {
    
    public:
    //! Constructor.
    RotationalController(std::shared_ptr<ros::NodeHandle>, int);
		
    void update();
    //! Destructor.
    ~RotationalController();
    
    private:
    
    //! Callback function for subscriber.
    void currentStateCallback(const nav_msgs::OdometryConstPtr &current);
    void nextStateCallback(const nav_msgs::OdometryConstPtr &next);
    ros::Subscriber current_state_sub, next_state_sub;
    ros::Publisher thruster_pub;
    //SG: why not an array ?
    int thrstr_1_spd, thrstr_2_spd, thrstr_3_spd, thrstr_4_spd, thrstr_5_spd, thrstr_6_spd;
    double current_yaw;
    double current_error_yaw = 0, previous_error_yaw = 0;
    double integral_error_yaw = 0, derivative_error_yaw = 0;
    double control_output;
    double Kp = 1, Kd = 1, Ki = 1, dt = 0.1;
};

#endif // ROTATIONAL_CONTROLLER_H

