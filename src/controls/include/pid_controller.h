#ifndef DEPTH_CONTROLLER_H
#define DEPTH_CONTROLLER_H

#include <ros/ros.h>
#include "std_msgs/Float64.h"


#include <dynamic_reconfigure/server.h>
#include <controls/TestConfig.h>

#include <boost/circular_buffer.hpp>

#define PI 3.14159


class PIDController {
    public:
    PIDController(ros::NodeHandle n, ros::NodeHandle np, std::string topic_name);
    ~PIDController();

	void update();
	
    protected:
    
    ros::Time m_prev_time; 

    ros::Subscriber m_sensor_sub;
    void sensorCallback(const std_msgs::Float64::ConstPtr& msg);
    double m_current;

	ros::Publisher m_command_pub;

    //void commandCallback(const std_msgs::Float64::ConstPtr& msg);
    std_msgs::Float64  m_command_msg;

	double m_desired;

    std::string m_control_topic;

	//used to average error terms, size determined in constructor
    boost::circular_buffer<double> m_error_buf;
	
	//P,I, and D terms. 
	double m_error_integral;
	double m_error;
	double m_error_derivative;
	
	double m_prev_error;

    //gains
    double m_kp;
    double m_ki;
    double m_kd;

    double m_upper_limit;
    double m_lower_limit;
    double m_windup_limit; 

    double m_cutoff_frequency;
    double m_update_frequency;

	bool m_unwind_angle; // tells us if our variable is an angle that we need to unwind. 


    //dynamic reconfigure stuff
	dynamic_reconfigure::Server<controls::TestConfig> server;
	dynamic_reconfigure::Server<controls::TestConfig>::CallbackType f;

    void configCallback(controls::TestConfig &config, uint32_t level);

	
};


#endif //PID_CONTROLLER_H


	
