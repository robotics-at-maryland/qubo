#include "controller_core.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "controller_node");
    std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle); 
    
    
    controlNode node(n, 0.5);
    
    while (ros::ok()){
        node.update();
        ros::Duration(0.1).sleep();
    }
    
}
