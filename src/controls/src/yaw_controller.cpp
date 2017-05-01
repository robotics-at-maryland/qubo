#include "yaw_controller.h"



using namespace std;
using namespace ros; 

YawController::YawController(NodeHandle n) {

	//TODO how accurate is ros::Time going to be for control purposes?
	m_prev_time = ros::Time::now();
	
    // Set up publishers and subscribers
	string qubo_namespace = "/qubo/";
	
	string sensor_topic = qubo_namespace + "yaw";
	m_sensor_sub = n.subscribe(sensor_topic, 1000, &YawController::sensorCallback, this);
	
	string cmd_topic = qubo_namespace + "yaw_cmd";
	m_cmd_pub = n.advertise<std_msgs::Float64>(cmd_topic, 1000);

	m_yaw_cmd.data = 5;
    
}

YawController::~YawController() {}


void YawController::update() {
	//update our commanded and measured yaw.
	ros::spinOnce();
	
	// Calculate time passed since previous loop
	ros::Duration dt = ros::Time::now() - m_prev_time;
	m_prev_time = ros::Time::now();
    
	//calculate error, update integrals and derivatives of the error
	m_error            = m_desired_yaw - m_current_yaw; //proportional term
	m_error_integral  += m_error * dt.toSec(); //integral term
	m_error_derivative = (m_error -m_prev_error)/dt.toSec();

	//store the previous error
	m_prev_error = m_error;

	//sum everything weighted by the given gains. 
	m_yaw_cmd.data = (m_kp*m_error) + (m_ki*m_error_integral) + (m_kd*m_error_derivative); 
	m_cmd_pub.publish(m_yaw_cmd);
	
}


void YawController::sensorCallback(const std_msgs::Float64::ConstPtr& msg) {
	m_current_yaw = msg->data;
	
}




