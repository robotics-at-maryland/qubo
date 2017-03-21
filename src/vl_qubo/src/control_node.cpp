#include "control_node.h"


using namespace std;
using namespace ros;

ControlNode::ControlNode(ros::NodeHandle n, string node_name, string ahrs_device, string imu_device)
    :_node_name{node_name}, _ahrs_device{ahrs_device}, _ahrs(ahrs_device,AHRS::k115200) {
        
        //initialize publisher
        _orientPub = n.advertise<sensor_msgs::Imu>("/qubo/orientation",1000);

        //creates + opens the device
        //k115200 is the current baud rate of the device, if you want to change it just change it right here it shouldn't show up anywhere else
        //        ahrs = AHRS(ahrs_device, AHRS::k115200);

        //Try to open the AHRS, an error if it doesn't work
        try{
            _ahrs.openDevice();
        }catch(AHRSException& ex){
            ROS_ERROR("%s", ex.what());
            return;
        }
        
        // make sure it's actually open
        if(!_ahrs.isOpen()){
            ROS_ERROR("AHRS %s didn't open succsesfully", _ahrs_device.c_str());
            return;
        }
        //configs the device
        _ahrs.sendAHRSDataFormat();
    
        ROS_DEBUG("Device Info: %s", _ahrs.getInfo().c_str());
    


        
        
}

ControlNode::~ControlNode(){
    _ahrs.closeDevice();
    //need to close IMU and Tiva connections too

    
}


void ControlNode::update(){}

void ControlNode::updateAHRS(){
    static int id = 0;
    static int attempts = 0;

	//if we aren't connected yet, lets try a few more times
	if(!_ahrs.isOpen()){
		try{
			_ahrs.openDevice();
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
		_ahrs_data = _ahrs.pollAHRSData();
	}catch(AHRSException& ex){
		//every so often an exception gets thrown and hangs on my vm
		//This might be solved by running ROS on actual hardware
		ROS_WARN("%s", ex.what());
		return;
	}

	//construct the imu data message
	ROS_DEBUG("Data has been read");
	_msg.header.stamp = ros::Time::now();
	_msg.header.seq = ++id;
	_msg.header.frame_id = "ahrs";

	_msg.orientation.x = _ahrs_data.quaternion[0];
	_msg.orientation.y = _ahrs_data.quaternion[1];
	_msg.orientation.z = _ahrs_data.quaternion[2];
	_msg.orientation.w = _ahrs_data.quaternion[3];
	// the -1's imply we don't know the covariance
	_msg.orientation_covariance[0] = -1;

	_msg.angular_velocity.x = _ahrs_data.gyroX;
	_msg.angular_velocity.y = _ahrs_data.gyroY;
	_msg.angular_velocity.z = _ahrs_data.gyroZ;
	_msg.angular_velocity_covariance[0] = -1;

	_msg.linear_acceleration.x = _ahrs_data.accelX;
	_msg.linear_acceleration.y = _ahrs_data.accelY;
	_msg.linear_acceleration.z = _ahrs_data.accelZ;
	_msg.linear_acceleration_covariance[0] = -1;

	//ahrsPub.publish(_msg);
    
}
