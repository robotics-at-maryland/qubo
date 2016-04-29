#include "thruster_tortuga.h"

ThrusterTortugaNode::ThrusterTortugaNode(int argc, char **argv, int rate): TortugaNode(){
    ros::Rate loop_rate(rate);
    subscriber = n.subscribe("/qubo/thruster_input", 1000, &ThrusterTortugaNode::thrusterCallBack, this);


    ROS_DEBUG("Opening sensorboard");
    sensor_file = "/dev/sensor";
    fd = openSensorBoard(sensor_file.c_str());
    ROS_DEBUG("Opened sensorboard with fd %d.", fd);
    checkError(syncBoard(fd));
    ROS_DEBUG("Synced with the Board");

    //GH: This is not the correct use of checkError.
    //It should be called on the return value of a function call, not on the file descriptor.
    //checkError(fd);

    // Unsafe all the thrusters
    ROS_DEBUG("Unsafing all thrusters");
    for (int i = 6; i <= 11; i++) {
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

ThrusterTortugaNode::~ThrusterTortugaNode(){
    //Stop all the thrusters
    ROS_DEBUG("Stopping thrusters");
    readSpeedResponses(fd);
    setSpeeds(fd, 0, 0, 0, 0, 0, 0);
    ROS_DEBUG("Safing thrusters");
    // Safe all the thrusters
    for (int i = 0; i <= 5; i++) {
        checkError(setThrusterSafety(fd, i));
    }
    ROS_DEBUG("Safed thrusters");
    //Close the sensorboard
    close(fd);
}

void ThrusterTortugaNode::update(){
    //I think we need to initialize thrusters and stuff before this will work
    ros::spinOnce();
    // setSpeeds(fd, msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5]);
    // ROS_DEBUG("fd = %x",fd);

    ROS_DEBUG("Setting thruster speeds");
    int retR = readSpeedResponses(fd);
    ROS_DEBUG("Read speed before: %x", retR);
    for(int i = 0 ;i < NUM_THRUSTERS; i++){
      ROS_DEBUG("thruster value %i = %i\n" i, thruster_powers.data[i])
    }
    int retS = setSpeeds(fd, thruster_powers.data[0], , thruster_powers.data[1], thruster_powers.data[2], thruster_powers[3], thruster_powers[4]);
    ROS_DEBUG("Set speed status: %x", retS);
    usleep(20*1000);
    int retA = readSpeedResponses(fd);
    ROS_DEBUG("Read speed after: %x", retA);

    ROS_DEBUG("thruster state = %x", readThrusterState(fd));
    ROS_DEBUG("set speed returns %x", retS);
    ROS_DEBUG("read speed returns %x", retR);
    
}

void ThrusterTortugaNode::publish(){
    // setSpeeds(fd, msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5]);
}

void ThrusterTortugaNode::thrusterCallBack(const std_msgs::Int64MultiArray new_powers){
  for(int i = 0 ;i < NUM_THRUSTERS; i++){
    thruster_powers.data[i] = new_powers.data[i];
  }
}

//ssh robot@192.168.10.12
