/** Nothing too fancy going on here, we just subscribe to uwsims "pressure" (which uwsim implements by
 *  just taking the depth and adding some noise)
 *  and publishing that to a topic.
 **/



#ifndef DEPTHSIM_HEADER
#define DEPTHSIM_HEADER

#include "ram_node.h"
#include "underwater_sensor_msgs/Pressure.h"

class DepthSimNode : public RamNode {
    
    public:
    DepthSimNode(std::shared_ptr<ros::NodeHandle> ,int);
    ~DepthSimNode();
    
    void update();
    void depthCallBack(const underwater_sensor_msgs::Pressure msg);
    
    protected:
    
    underwater_sensor_msgs::Pressure msg;
    ros::Subscriber subscriber;
    ros::Publisher publisher;
    
};



#endif
