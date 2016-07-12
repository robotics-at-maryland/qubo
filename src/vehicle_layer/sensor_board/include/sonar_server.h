#ifndef SONAR_SERVER_H
#define SONAR_SERVER_H


//SG: change this ASAP
#include "ros/ros.h"
#include "ram_msgs/sonar_data.h"
#include "sensor_board_tortuga.h"
#include "sensorapi.h"


class SonarServerNode : public SensorBoardTortugaNode {
    
    public:
    SonarServerNode(std::shared_ptr<ros::NodeHandle>, int rate, int fd ,  std::string file_name);
    ~SonarServerNode();
    
    void update();
    //void thrusterCallBack(const std_msgs::Int64MultiArray msg);
    
    protected:
    //contains the current powers we want the thrusters to be operating at
    //this is the DESIRED relative power, since our thrusters our nonlinear
    //we'll need to map these to another vector eventually.
    //std_msgs::Int64MultiArray thruster_powers;
    sonarData sd;
    int fd;
    std::string sensor_file;
    ros::ServiceServer service; 
};

#endif
