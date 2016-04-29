#ifndef SENSOR_BOARD_TORTUGA_H
#define SENSOR_BOARD_TORTUGA_H

#include "tortuga_node.h"
#include "tortuga/sensorapi.h"

/*
 * Dependencies for thrusters  
 */
#include "std_msgs/Int64MultiArray.h"
#include "sensor_msgs/Joy.h"

class SensorBoardTortugaNode : public TortugaNode {
	public:
        SensorBoardTortugaNode(int, char**, int);
        ~SensorBoardTortugaNode();
        
        void update();
        void publish();
        
        // Thruster functions
        void updateThrusters();
        void thrusterCallBack(const std_msgs::Int64MultiArray msg);

    protected:
    int sensor_fd; //File descriptor for sensor board
    std::string sensor_file; // Serial port/system location for sensor board
    // Thruster Variables
    std_msgs::Int64MultiArray thruster_msg; // Contains thruster commands
    ros::Subscriber thruster_subscriber; // Reads thruster commands from thruster_msg

};

#endif
