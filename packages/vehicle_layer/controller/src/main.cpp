#include "controller_core.h"

int main(int argc, char **argv) {
  std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle); 
    ros::init(argc, argv, "controller_node");
    
    //TODO make work for tortuga node?
    std::unique_ptr<RamNode> node;
    node.reset(new controlNode(n, 10));
    
    while (ros::ok()){
        node->update();
    }
    
}
