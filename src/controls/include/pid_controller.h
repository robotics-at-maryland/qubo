#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>

namespace Controls {

class PIDController {
    public:
        PIDController(ros::NodeHandle nh);
        ~PIDController();
        void run();
        void robot_state_callback(const nav_msgs::OdometryConstPtr& current_state);
        void desired_state_callback(const nav_msgs::OdometryConstPtr& desired_state);
        
    private:
        static constexpr float upper_limit = 1000.0;
        static constexpr float lower_limit = -1000.0;
        static constexpr float Kp_x = 0.0;
        static constexpr float Ki_x = 0.0;
        static constexpr float Kd_x = 0.0;
        static constexpr float Kp_y = 0.0;
        static constexpr float Ki_y = 0.0;
        static constexpr float Kd_y = 0.0;
        static constexpr float Kp_z = 0.0;
        static constexpr float Ki_z = 0.0;
        static constexpr float Kd_z = 0.0;
    
        float desired_x, desired_y, desired_z;
        float error_x, error_y, error_z;
        float integral_error_x, integral_error_y, integral_error_z;
        float prev_error_x, prev_error_y, prev_error_z;

        ros::Time prev_time;

        ros::Subscriber robot_state_sub;
        ros::Subscriber desired_state_sub;
};

} // namespace Controls

#endif //PID_CONTROLLER_H
