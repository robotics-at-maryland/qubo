#include "sensor_board_tortuga.h"

SensorBoardTortugaNode::SensorBoardTortugaNode(int argc, char **argv, int rate): TortugaNode() {
    ros::Rate loop_rate(rate);
    
    ROS_DEBUG("Opening sensor board");
    sensor_file = "/dev/sensor";
    sensor_fd = openSensorBoard(sensor_file.c_str()); // Function from sensorapi.h
    ROS_DEBUG("Opened sensor board with fd %d.", sensor_fd);
    checkError(syncBoard(sensor_fd)); // Function from sensorapi.h
    ROS_DEBUG("Synced with the Board");

    // Thruster initiatization
    thruster_subscriber = n.subscribe("/qubo/thruster_input", 1000, &SensorBoardTortugaNode::thrusterCallBack, this);
    ROS_DEBUG("Unsafing all thrusters");
    /*
     * Following code unsafes all thrusters...
     * 
     * i = [0 to 5] are safing corresponding thrusters
     * i = [6 to 11] are unsafing corresponding thrusters
     *
     * e.g. If i = 0, first thruster is safed. If i = 6, first thruster is unsafed.
     */
    for(int i = 6; i <= 11; i++) {
        checkError(setThrusterSafety(sensor_fd, i));
    }
    ROS_DEBUG("Unsafed all thrusters");
    
    // Temperature initialization
    temperature_publisher = n.advertise<ram_msgs::Temperature>("qubo/temp/", 1000);
}

SensorBoardTortugaNode::~SensorBoardTortugaNode() {
    // Thruster destruction
    ROS_DEBUG("Stopping thrusters");
    readSpeedResponses(sensor_fd);
    setSpeeds(sensor_fd, 0, 0, 0, 0, 0, 0) ;
    // Safe all thrusters
    for(int i = 0; i <= 5 ; i++) {
        checkError(setThrusterSafety(sensor_fd, i));
    }
    ROS_DEBUG("Safed thrusters");
    // Close the sensor board
    close(sensor_fd);
}

void SensorBoardTortugaNode::update() {
    ros::spinOnce();
    updateThrusters();
    updateTemperature();
} 

void SensorBoardTortugaNode::publish() {
    // NOT USED
}

void SensorBoardTortugaNode::updateThrusters() {
    ROS_DEBUG("Setting thruster speeds");
    int retR = readSpeedResponses(sensor_fd);
    ROS_DEBUG("Read speed before: %x", retR);
    int retS = setSpeeds(sensor_fd, 128, 128, 128, 128, 128, 128);
    ROS_DEBUG("Set speed status: %x", retS);
    usleep(20*1000);
    int retA = readSpeedResponses(sensor_fd);
    ROS_DEBUG("Read speed after: %x", retA);

    ROS_DEBUG("    thruster state = %x", readThrusterState(sensor_fd));
    ROS_DEBUG("    set speed returns %x", retS);
    ROS_DEBUG("    read speed returns %x", retR);
}

void SensorBoardTortugaNode::updateTemperature() {
    unsigned char temps[NUM_TEMP_SENSORS];
    readTemp(sensor_fd, temps);
    temp_msg.temp1 = temps[0];
}

void SensorBoardTortugaNode::thrusterCallBack(const std_msgs::Int64MultiArray new_vector) {
    thruster_msg.data = new_vector.data;
}
