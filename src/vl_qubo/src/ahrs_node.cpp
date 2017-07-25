#include "ahrs_node.h"


using namespace std;
using namespace ros;

AHRSQuboNode::AHRSQuboNode(ros::NodeHandle n, string node_name, string ahrs_device)
    :m_node_name{node_name}, m_ahrs_device{ahrs_device}, m_ahrs(ahrs_device,AHRS::k115200) {


		m_roll_pub  = n.advertise<std_msgs::Float64>("/qubo/roll", 1000);
		m_pitch_pub = n.advertise<std_msgs::Float64>("/qubo/pitch", 1000);
		m_yaw_pub   = n.advertise<std_msgs::Float64>("qubo/yaw", 1000);
	

        //initialize publisher
		//        m_orientPub = n.advertise<sensor_msgs::Imu>("/qubo/orientation",1000);
		ROS_ERROR("opening device %s", ahrs_device.c_str());
        //creates + opens the device
        //k115200 is the current baud rate of the device, if you want to change it just change it right here it shouldn't show up anywhere else
        //        ahrs = AHRS(ahrs_device, AHRS::k115200);

        //Try to open the AHRS, an error if it doesn't work
        try{
            m_ahrs.openDevice();
        }catch(AHRSException& ex){
            ROS_ERROR("%s", ex.what());
            return;
        }
        
        // make sure it's actually open
        if(!m_ahrs.isOpen()){
            ROS_ERROR("AHRS %s didn't open succsesfully", m_ahrs_device.c_str());
            return;
        }
        //configs the device
        m_ahrs.sendAHRSDataFormat();
    
        ROS_DEBUG("Device Info: %s", m_ahrs.getInfo().c_str());
    
        
}

AHRSQuboNode::~AHRSQuboNode(){
    m_ahrs.closeDevice();

    
}

void AHRSQuboNode::update(){
    static int id = 0;
    static int attempts = 0;

	//if we aren't connected yet, lets try a few more times
	if(!m_ahrs.isOpen()){
		try{
			m_ahrs.openDevice();
		}catch(AHRSException& ex){
			ROS_ERROR("Attempt %i to connect to AHRS failed.", attempts++);
			ROS_ERROR("DEVICE NOT FOUND! ");
			ROS_ERROR("%s", ex.what());
			if(attempts > _MAX_CONNECTION_ATTEMPTS){
				ROS_ERROR("Failed to find device, exiting node.");
				exit(-1);
			}
			return;
		}
	}
	attempts = 0;

	ROS_DEBUG("Beginning to read data");
	//sit and wait for an update
	try{
		m_ahrs_data = m_ahrs.pollAHRSData();
	}catch(AHRSException& ex){
		//every so often an exception gets thrown and hangs on my vm
		//This might be solved by running ROS on actual hardware
		ROS_WARN("%s", ex.what());
		return;
	}

	//construct the imu data message
	/// /	ROS_ERROR("Data has been read");
	// m_msg.header.stamp = ros::Time::now();
	// m_msg.header.seq = ++id;
	// m_msg.header.frame_id = "ahrs";

	// m_msg.orientation.x = m_ahrs_data.quaternion[0];
	// m_msg.orientation.y = m_ahrs_data.quaternion[1];
	// m_msg.orientation.z = m_ahrs_data.quaternion[2];
	// m_msg.orientation.w = m_ahrs_data.quaternion[3];
   

	//this is a little clunky, but it's the best way I could find to convert from a quaternion to Euler Angles
	tf::Quaternion q(m_ahrs_data.quaternion[0], m_ahrs_data.quaternion[1], m_ahrs_data.quaternion[2], m_ahrs_data.quaternion[3]);
	tf::Matrix3x3 m(q);
	
	m.getRPY(m_roll_msg.data, m_pitch_msg.data, m_yaw_msg.data); //roll pitch and yaw are populated

	ROS_ERROR("%f, %f, %f", m_roll_msg.data, m_pitch_msg.data, m_yaw_msg.data);

	m_roll_pub.publish(m_roll_msg);
	m_pitch_pub.publish(m_pitch_msg);
	m_yaw_pub.publish(m_yaw_msg);
	

	
	// the -1's imply we don't know the covariance
	// m_msg.orientation_covariance[0] = -1;

	// m_msg.angular_velocity.x = m_ahrs_data.gyroX;
	// m_msg.angular_velocity.y = m_ahrs_data.gyroY;
	// m_msg.angular_velocity.z = m_ahrs_data.gyroZ;
	// m_msg.angular_velocity_covariance[0] = -1;

	// m_msg.linear_acceleration.x = m_ahrs_data.accelX;
	// m_msg.linear_acceleration.y = m_ahrs_data.accelY;
	// m_msg.linear_acceleration.z = m_ahrs_data.accelZ;
	// m_msg.linear_acceleration_covariance[0] = -1;

	// //ahrsPub.publish(_msg);
    
}
