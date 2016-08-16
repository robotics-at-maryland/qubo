#ifndef THRUSTER_TORTUGA_H
#define THRUSTER_TORTUGA_H

/* Thruster Node class to be run on tortuga, requires someone else to open up the sensor board and pass it in, this is
   meant to make sharing said sensor board much easier
*/

#define NUM_THRUSTERS 6

#include "sensor_board_tortuga.h"
#include "std_msgs/Int64MultiArray.h"
#include "sensorapi.h"


class ThrusterTortugaNode : public SensorBoardTortugaNode {
    
    public:
    ThrusterTortugaNode(std::shared_ptr<ros::NodeHandle>, int rate, int fd ,  std::string file_name);
    ~ThrusterTortugaNode();
    
    void update();
    void thrusterCallBack(const std_msgs::Int64MultiArray msg);
    
    protected:
    //contains the current powers we want the thrusters to be operating at
    //this is the DESIRED relative power, since our thrusters our nonlinear
    //we'll need to map these to another vector eventually.
    std_msgs::Int64MultiArray thruster_powers;
    int fd;
    std::string sensor_file;
    ros::Subscriber subscriber;
    
};

#endif
