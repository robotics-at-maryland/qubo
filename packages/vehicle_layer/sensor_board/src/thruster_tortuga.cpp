 #include "thruster_tortuga.h"

ThrusterTortugaNode::ThrusterTortugaNode(std::shared_ptr<ros::NodeHandle> n, int rate, int board_fd, std::string board_file):
    SensorBoardTortugaNode(n, rate, board_fd, board_file) {

    //SG: should we make qubo/thruster_input a variable or macro? yes, but this is the only place it shows up, we're going to need
    //to come back and change it anyway, and I don't want to fix it so there.
    subscriber = n->subscribe("/qubo/thruster_input", 1000, &ThrusterTortugaNode::thrusterCallBack, this);
    

    ROS_DEBUG("Unsafing all thrusters");
    for(int i = 6; i <= 11; i++) {
        checkError(setThrusterSafety(fd, i));
    }
	
    ROS_DEBUG("Unsafed all thrusters");
    thruster_powers.layout.dim.push_back(std_msgs::MultiArrayDimension());
    thruster_powers.layout.dim[0].size = 6;
    thruster_powers.data.reserve(thruster_powers.layout.dim[0].size);
    
    for(int i = 0; i < NUM_THRUSTERS; i++){
        thruster_powers.data[i] = 0;
    }

}

void ThrusterTortugaNode::update(){
    //I think we need to initialize thrusters and stuff before this will work
    ros::spinOnce();
   

    // ROS_DEBUG("Setting thruster speeds");
    int retR = readSpeedResponses(fd);
    // ROS_DEBUG("Read speed before: %x", retR);

    int retS = setSpeeds(fd, thruster_powers.data[0], thruster_powers.data[1], thruster_powers.data[2], thruster_powers.data[3], thruster_powers.data[4], thruster_powers.data[5]);
    //ROS_DEBUG("Set speed status: %x", retS);
    usleep(20*1000);
    int retA = readSpeedResponses(fd);
    //ROS_DEBUG("Read speed after: %x", retA);

    //ROS_DEBUG("thruster state = %x", readThrusterState(fd));
    // ROS_DEBUG("set speed returns %x", retS);
    //ROS_DEBUG("read speed returns %x", retR);
   
}

void ThrusterTortugaNode::thrusterCallBack(const std_msgs::Int64MultiArray new_powers){
  for(int i = 0 ;i < NUM_THRUSTERS; i++){
      ROS_DEBUG("received a message!");
    thruster_powers.data[i] = new_powers.data[i];


    update();
  }
}
