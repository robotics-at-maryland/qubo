#include "thruster_tortuga.h"

ThrusterTortugaNode::ThrusterTortugaNode(int argc, char **argv, int rate): TortugaNode(){
    ros::Rate loop_rate(rate);
    subscriber = n.subscribe("/qubo/thruster_input", 1000, &ThrusterTortugaNode::thrusterCallBack, this);
 

    printf("Opening sensorboard\n");
    sensor_file = "/dev/sensor";
    fd = openSensorBoard(sensor_file.c_str());
    printf("Opened sensorboard with fd %d.\n", fd);
    checkError(syncBoard(fd));
    printf("Synced with the Board\n");

    //GH: This is not the correct use of checkError.
    //It should be called on the return value of a function call, not on the file descriptor.
    //checkError(fd);

    // Unsafe all the thrusters
    printf("Unsafing all thrusters\n");
    for (int i = 6; i <= 11; i++) {
        checkError(setThrusterSafety(fd, i));
    }
    printf("Unsafed all thrusters\n");
  
}

ThrusterTortugaNode::~ThrusterTortugaNode(){
    //Stop all the thrusters
    printf("Stopping thrusters\n");
    readSpeedResponses(fd);
    setSpeeds(fd, 0, 0, 0, 0, 0, 0);
    printf("Safing thrusters\n");
    // Safe all the thrusters
    for (int i = 0; i <= 5; i++) {
        checkError(setThrusterSafety(fd, i));
    }
    printf("Safed thrusters\n");
    //Close the sensorboard
    fclose(fd);
}

void ThrusterTortugaNode::update(){
    //I think we need to initialize thrusters and stuff before this will work 
    ros::spinOnce();
    // setSpeeds(fd, msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5]);
    // printf("fd = %x\n",fd); 
   
    printf("Setting thruster speeds\n");
    int retR = readSpeedResponses(fd);
    printf("Read speed before: %x\n", retR);
    int retS = setSpeeds(fd, 128, 128, 128, 128, 128, 128);
    printf("Set speed status: %x\n", retS);
    usleep(20*1000);
    int retA = readSpeedResponses(fd);
    printf("Read speed after: %x\n", retA);
      
    ROS_ERROR("thruster state = %x\n", readThrusterState(fd)); 
    ROS_ERROR("set speed returns %x\n", retS);
    ROS_ERROR("read speed returns %x\n", retR);
}

void ThrusterTortugaNode::publish(){
    // setSpeeds(fd, msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5]);
}

void ThrusterTortugaNode::thrusterCallBack(const std_msgs::Float64MultiArray new_vector){
    //SG: TODO change this shit
    msg.data = new_vector.data;
}

//ssh robot@192.168.10.12
