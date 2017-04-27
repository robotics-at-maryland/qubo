#include "pid_controller.h"

namespace Controls {

PIDController::PIDController(ros::NodeHandle *nh) {
    // Initialize all instance variables to 0
    desired_x = desired_y = desired_z = desired_t = 0.0;
    integral_error_x = integral_error_y = integral_error_z = integral_error_t = 0.0;
    prev_error_x = prev_error_y = prev_error_z = prev_error_t = 0.0;
    prev_time = ros::Time::now();

    // Set up publishers and subscribers
    robot_state_sub = nh->subscribe("/qubo/current_state", 1, &PIDController::robot_state_callback, this);
    desired_state_sub = nh->subscribe("/qubo/desired_state", 1, &PIDController::desired_state_callback, this);
    thruster_pub = nh->advertise<std_msgs::Int64MultiArray>("/qubo/thruster_input", 1);
}

PIDController::~PIDController() {}

void PIDController::run() {
    ros::spin();
}

void PIDController::robot_state_callback(const nav_msgs::OdometryConstPtr& current_state) {
    // Calculate time passed since previous loop
    ros::Duration dt = ros::Time::now() - prev_time;
    prev_time = ros::Time::now();
    
    // Calculate proportional error
    double error_x = desired_x - current_state->pose.pose.position.x;
    double error_y = desired_y - current_state->pose.pose.position.y;
    double error_z = desired_z - current_state->pose.pose.position.z;
    double error_t = desired_t - tf::getYaw(current_state->pose.pose.orientation);
    
    // Calculate integral error
    integral_error_x += error_x * dt.toSec();
    integral_error_y += error_y * dt.toSec();
    integral_error_z += error_z * dt.toSec();
    integral_error_t += error_t * dt.toSec();
    
    // Calculate derivative error
	//sgillen@20172512-13:25 - may need to filter this, or use really small D terms..
    double derivative_error_x = (error_x - prev_error_x) / dt.toSec();
    double derivative_error_y = (error_y - prev_error_y) / dt.toSec();
    double derivative_error_z = (error_z - prev_error_z) / dt.toSec();
    double derivative_error_t = (error_t - prev_error_t) / dt.toSec();

    // Save previous errors
    prev_error_x = error_x;
    prev_error_y = error_y;
    prev_error_z = error_z;
    prev_error_t = error_t;
    
    // Combine errors to get total control effort
    double control_effort_x = Kp_x * error_x + Ki_x * integral_error_x + Kd_x * derivative_error_x;
    double control_effort_y = Kp_y * error_y + Ki_y * integral_error_y + Kd_y * derivative_error_y;
    double control_effort_z = Kp_z * error_z + Ki_z * integral_error_z + Kd_z * derivative_error_z;
    double control_effort_t = Kp_t * error_t + Ki_t * integral_error_t + Kd_t * derivative_error_t;
    
    // Apply upper limit to control efforts
    if (control_effort_x > upper_limit) {
        control_effort_x = upper_limit;
    } else if (control_effort_x < lower_limit) {
        control_effort_x = lower_limit;
    }
    
    if (control_effort_y > upper_limit) {
        control_effort_y = upper_limit;
    } else if (control_effort_y < lower_limit) {
        control_effort_y = lower_limit;
    }
    
    if (control_effort_z > upper_limit) {
        control_effort_z = upper_limit;
    } else if (control_effort_z < lower_limit) {
        control_effort_z = lower_limit;
    }

    if (control_effort_t > upper_limit) {
        control_effort_t = upper_limit;
    } else if (control_effort_t < lower_limit) {
        control_effort_t = lower_limit;
    }
    
    /* Publish control efforts to thrusters. TBD because thrusters don't exist
     * yet... */
}

void PIDController::desired_state_callback(const nav_msgs::OdometryConstPtr& desired_state) {
    desired_x = desired_state->pose.pose.position.x;
    desired_y = desired_state->pose.pose.position.y;
    desired_z = desired_state->pose.pose.position.z;
    desired_t = tf::getYaw(desired_state->pose.pose.orientation);
}

} // namespace Controls
