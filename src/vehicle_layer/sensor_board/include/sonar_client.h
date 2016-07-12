#ifndef SONAR_CLIENT_H
#define SONAR_CLIENT_H


//SG: change this ASAP
#include "ros/ros.h"
#include <ram_msgs/sonar_data.h>
#include "sensor_board_tortuga.h"
#include "sensorapi.h"
#include <cstdlib>

class SonarClientNode : public SensorBoardTortugaNode {
    
    public:
    SonarClientNode(std::shared_ptr<ros::NodeHandle>, int rate, int fd ,  std::string file_name);
    ~SonarServerNode();
    
    //void update();
    //void thrusterCallBack(const std_msgs::Int64MultiArray msg);
    
    protected:
    //contains the current powers we want the thrusters to be operating at
    //this is the DESIRED relative power, since our thrusters our nonlinear
    //we'll need to map these to another vector eventually.
    //std_msgs::Int64MultiArray thruster_powers;
    //int fd;
    ram_msgs::sonar_data srv;
    std::string sensor_file;
    ros::ServiceClient client; 
};

#endif
