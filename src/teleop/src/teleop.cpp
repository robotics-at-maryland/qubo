#include <ros/ros.h>
// INCLUDE FLOAT64MULTIARRAY
#include <sensor_msgs/Joy.h>

class TeleopNode {
    public:
        TeleopNode();

    private:
        void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);

        ros::NodeHandle nh_;

        int linear_, angular_;
        double l_scale_, a_scale_;
        ros::Publisher vel_pub_;
        ros::Subscriber joy_sub_;
};

TeleopNode::TeleopNode():
    linear_(1),
    angular_(2)
{
    vel_pub_ = nh_.advertise<***FLOAT64MULTIARRAY***>("/tortuga/thruster_input", 1);
