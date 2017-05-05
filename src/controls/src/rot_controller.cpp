#include "rot_controller.h"

#define PI 3.14159

using namespace std;
using namespace ros;

//this calls the PID constructor 
RotController::RotController(NodeHandle n, string control_topic)
	:PIDController(n, control_topic){}

RotController::~RotController(){};

void RotController::update(){
	//update our commanded and measured depth.
	ros::spinOnce();
	
	// Calculate time passed since previous loop
	ros::Duration dt = ros::Time::now() - m_prev_time;
	m_prev_time = ros::Time::now();
    
	//calculate error, update integrals and derivatives of the error
	m_error            = m_desired  - m_current; //proportional term

	//makes sure we always take the smallest way around the circle
	if(m_error > PI){
		m_error = 2*PI - m_error;
	}else if(m_error < -PI){
		m_error = 2*PI + m_error;
	}

	
	m_error_integral  += m_error * dt.toSec(); //integral term
	m_error_derivative = (m_error - m_prev_error)/dt.toSec();

	//store the previous error
	m_prev_error = m_error;

	ROS_INFO("%s: ep = %f ei = %f ed = %f, dt = %f", m_control_topic.c_str(), m_error, m_error_integral, m_error_derivative,  dt.toSec());  
	//sum everything weighted by the given gains. 
	m_command_msg.data = (m_kp*m_error) + (m_ki*m_error_integral) + (m_kd*m_error_derivative); 
	m_command_pub.publish(m_command_msg);
}
