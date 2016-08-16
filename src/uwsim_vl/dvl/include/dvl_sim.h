#ifndef DVL_SIM_HEADER
#define DVL_SIM_HEADER

#include "ram_node.h"
#include "underwater_sensor_msgs/DVL.h"

class DVLSimNode : public RamNode {

 public:
    DVLSimNode(std::shared_ptr<ros::NodeHandle> , int);
    ~DVLSimNode();
    
    void update();
    void dvlCallBack(const underwater_sensor_msgs::DVL msg);
    
    protected:
    
    underwater_sensor_msgs::DVL msg;
    ros::Subscriber subscriber;
    ros::Publisher publisher;
    

};

#endif
