//sgillen@20172214-15:22 really just a simple PID controller, subscribes to a given sensor topic,
//a given goal topic (IE what we want the sensor to read) and publishes a given command effort topic
//ros already has one of these built in but I wanted more control over what was going on to filter
//with a simple average and to give the option of subscribing to an error term directly
//(though that is not implemented yet)  


#include "pid_controller.h"

using namespace std;
using namespace ros;


PIDController::PIDController(NodeHandle n, NodeHandle np,  string control_topic):
	m_control_topic(control_topic){


	int buf_size; //don't need this anywhere else might as well keep it local
	// Get params if specified in launch file or as params on command-line, set defaults
	np.param<double>("kp", m_kp, 1.0);
	np.param<double>("ki", m_ki, 0.0);
	np.param<double>("kd", m_kd, 0.0);
	np.param<double>("upper_limit", m_upper_limit, 1000.0);
	np.param<double>("lower_limit", m_lower_limit, -1000.0);
	np.param<double>("windup_limit", m_windup_limit, 1000.0);
	np.param<bool>("angular_variable" , m_unwind_angle, false);
	np.param<int>("buffer_size", buf_size, 1);

	m_error_buf.resize(buf_size);

	//TODO reconfigure bounds for the angle

	m_prev_time = ros::Time::now();
	
	//TODO have this be configurable?
	string qubo_namespace = "/qubo/";
	
	// Set up publishers and subscribers
	string sensor_topic = qubo_namespace + control_topic;
	m_sensor_sub = n.subscribe(sensor_topic, 1000, &PIDController::sensorCallback, this);

	string target_topic = qubo_namespace + control_topic + "_target";
	m_target_sub = n.subscribe(target_topic, 1000, &PIDController::targetCallback, this);
	
	string command_topic = qubo_namespace + control_topic + "_cmd";
	m_command_pub = n.advertise<std_msgs::Float64>(command_topic, 1000);
		
	//m_f =
	m_server.setCallback( boost::bind(&PIDController::configCallback, this, _1, _2));
	
    
}

PIDController::~PIDController(){}


void PIDController::update() {
	//update our commanded and measured depth.
	ros::spinOnce();

	// Calculate time passed since previous loop
	ros::Duration dt = ros::Time::now() - m_prev_time;
	m_prev_time = ros::Time::now();


	//proportional term
	//------------------------------------------------------------------------------
	//calculate error, update integrals and derivatives of the error
	m_error = m_target  - m_current; //proportional term

   	//if we are told to unwind our angle then we better do that. 
	if(m_unwind_angle){
		//makes sure we always take the smallest way around the circle
		if(m_error > PI){
			m_error = 2*PI - m_error;
		}else if(m_error < -PI){
			m_error = 2*PI + m_error;
		}
	}

	//add the newest error term to our buffer of terms so we can take the average
	m_error_buf.push_back(m_error);
	
	double sum = 0; //add the buffer up and divide to get the current smoothed error term
	for(int i = 0; i < m_error_buf.size(); i++){
		//ROS_ERROR("buf[%i] = %f", i, m_error_buf[i]);
		sum += m_error_buf[i];
	}

	m_error = sum/m_error_buf.size();
	

	
	//------------------------------------------------------------------------------
	//compute the integral and limit it if it's too big 
	m_error_integral  += m_error * dt.toSec(); //integral term
	
	//if the integral value is past our windup limit just set it there. 
	if(m_error_integral > m_windup_limit){
		m_error_integral = m_windup_limit;
	}
	else if(m_error_integral < -m_windup_limit){
		m_error_integral = -m_windup_limit;
	}


	//derivative term
	//------------------------------------------------------------------------------
	m_error_derivative = (m_error - m_prev_error)/dt.toSec();//smoothing is done by averaging the error, which we've already done 


	
	//sum everything weighted by the given gains.
	//------------------------------------------------------------------------------
	//ROS_INFO("%s: ep = %f ei = %f ed = %f, dt = %f", m_control_topic.c_str(), m_error,  m_error_integral, m_error_derivative, dt.toSec());  
	m_command_msg.data = (m_kp*m_error) + (m_ki*m_error_integral) + (m_kd*m_error_derivative);

	//make sure our error term is within limits
	if(m_command_msg.data > m_upper_limit){
		m_command_msg.data = m_upper_limit;
	}
	else if(m_command_msg.data < m_lower_limit){
		m_command_msg.data = m_lower_limit;
	}

	//publish the final result
	m_command_pub.publish(m_command_msg);
	
}

void PIDController::sensorCallback(const std_msgs::Float64::ConstPtr& msg) {
	m_current = msg->data;
	
}


void PIDController::configCallback(controls::TestConfig &config, uint32_t level) {
	ROS_INFO("Reconfigure Request: %f %f %f %f", config.kp, config.ki, config.kd, config.target);
	m_kp = config.kp;
	m_ki = config.ki;
	m_kd = config.kd;
	m_target = config.target;


	
	m_error = 0;
	m_error_integral = 0; //reset the integral error every time we switch things up
	m_error_derivative = 0;
	
}

void PIDController::targetCallback(const std_msgs::Float64::ConstPtr& msg) {
	m_target = msg->data;
}


