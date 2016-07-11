#include "dvl_tortuga.h"

DVLTortugaNode::DVLTortugaNode(std::shared_ptr<ros::NodeHandle> n, int rate, int board_fd, std::string board_file):RamNode(n) {
    
    publisher = n->advertise<geometry_msgs::TwistWithCovarianceStamped>("tortuga/dvl", 1000);
	ROS_DEBUG("Set up publisher");
    fd = board_fd;
    file = board_file;
    
    //intitialize all the message fields so no one gets upset if we send a message before getting good data. 
    msg.header.frame_id = "base_link";
	msg.header.stamp = ros::Time::now();
    
    msg.twist.twist.linear.x = 0;
    msg.twist.twist.linear.y = 0;
    msg.twist.twist.linear.z = 0;
    
	msg.twist.covariance = {1e-9, 0, 0, 0, 0, 0,
                            0, 1e-9, 0, 0, 0, 0,
                            0, 0, 1e-9, 0, 0, 0,
                            0, 0, 0, 1e-9, 0, 0,
                            0, 0, 0, 0, 1e-9, 0,
                            0, 0, 0, 0, 0, 1e-9};
    
   
}

DVLTortugaNode::~DVLTortugaNode(){}

void DVLTortugaNode::update(){
//	ros::spinOnce();

    checkError(readDVLData(fd, &raw));
	ROS_DEBUG("Read DVL Data");
    // Raw data has a pointer to complete packet
    pkt = raw.privDbgInf;
    if(raw.xvel_btm == DVL_BAD_DATA || raw.yvel_btm == DVL_BAD_DATA || raw.zvel_btm == DVL_BAD_DATA){
        ROS_ERROR("Bad Data, publishing last good value");
        publisher.publish(msg);
    }
    
    else{
        
    	// Set all the message's data
        msg.header.frame_id = "base_link";
        msg.header.stamp = ros::Time::now();
        
    	msg.twist.twist.linear.x = raw.xvel_btm;
    	msg.twist.twist.linear.y = raw.yvel_btm;
    	msg.twist.twist.linear.z = raw.zvel_btm;
        
        msg.twist.covariance = {1e-9, 0, 0, 0, 0, 0,
                                0, 1e-9, 0, 0, 0, 0,
                                0, 0, 1e-9, 0, 0, 0,
                                0, 0, 0, 1e-9, 0, 0,
                                0, 0, 0, 0, 1e-9, 0,
                                0, 0, 0, 0, 0, 1e-9};
        
    	publisher.publish(msg);
        
        ROS_DEBUG("Published msg");
        
    }
}

bool DVLTortugaNode::checkError(int e) {
    switch(e) {
    case ERR_NOSYNC:
        ROS_ERROR("NOSYNC ERROR in node %s", file.c_str());
        return true;
    case ERR_TOOBIG:
        ROS_ERROR("TOOBIG ERROR in node %s", file.c_str());
      return true;
    case ERR_CHKSUM:
        ROS_ERROR("CHKSUM ERROR in node %s", file.c_str());
        return true;
    default:
        return false;
    }
}


