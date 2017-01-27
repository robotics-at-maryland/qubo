#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int64MultiArray.h>

namespace Controls {

/*
 * Basic implementation of a PID Controller for four degrees of freedom (x, y,
 * z, theta).
 */
class PIDController {
    public:
        /* 
         * Constructor. Accepts a node handle to set up subscribers.
         */
        PIDController(ros::NodeHandle *nh);

        /* 
         * Destructor.
         */
        ~PIDController();

        /*
         * Call this method after the PID Controller has been constructed to
         * run the controller.
         */
        void run();

        /*
         * Callback method for the current state of the robot. The actual PID
         * loop computations are performed in this method.
         */
        void robot_state_callback(const nav_msgs::OdometryConstPtr& current_state);

        /*
         * Callback method for the desired state of the robot. Simply stored
         * the given desired state in instance variables.
         */
        void desired_state_callback(const nav_msgs::OdometryConstPtr& desired_state);
        
    private:
        // Kp, Ki, and Kd terms for x, y, and z
        static constexpr double Kp_x = 0.0;
        static constexpr double Ki_x = 0.0;
        static constexpr double Kd_x = 0.0;
        static constexpr double Kp_y = 0.0;
        static constexpr double Ki_y = 0.0;
        static constexpr double Kd_y = 0.0;
        static constexpr double Kp_z = 0.0;
        static constexpr double Ki_z = 0.0;
        static constexpr double Kd_z = 0.0;
        static constexpr double Kp_t = 0.0;
        static constexpr double Ki_t = 0.0;
        static constexpr double Kd_t = 0.0;

        // Upper and lower limits for control effort
        static constexpr double upper_limit = 1000.0;
        static constexpr double lower_limit = -1000.0;
   
        // Most recently received desired state for the robot 
        double desired_x, desired_y, desired_z, desired_t;

        // Integral and previous errors need to persist between iterations
        double integral_error_x, integral_error_y, integral_error_z, integral_error_t;
        double prev_error_x, prev_error_y, prev_error_z, prev_error_t;
        
        // Time of last iteration, need to keep track of this for calculating
        // derivative term.
        ros::Time prev_time;

        // ROS subscribers for robot's current state and goal state
        ros::Subscriber robot_state_sub;
        ros::Subscriber desired_state_sub;

        // ROS publisher for sending input to thrusters
        ros::Publisher thruster_pub;
};

} // namespace Controls

#endif //PID_CONTROLLER_H
