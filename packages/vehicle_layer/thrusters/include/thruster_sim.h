//! This will serve as the simulated version of our thrusters


#ifndef THRUSTER_SIM_HEADER //I don't really see anybody needing to inherit this class, but better safe than sorry. 
#define THRUSTER_SIM_HEADER

#include "ram_node.h"
#include "std_msgs/Float64MultiArray.h"

class ThrusterSimNode : public RamNode {
    
    public:
    ThrusterSimNode(std::shared_ptr<ros::NodeHandle>  ,int);
    ~ThrusterSimNode();
    
    void update();
    void publish();
    void thrusterCallBack(const std_msgs::Float64MultiArray msg);
    void cartesianToVelocity(double*);
  
    protected:
    std_msgs::Float64MultiArray msg;
    ros::Subscriber subscriber;
};



#endif
