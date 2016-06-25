#include "controller_core.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "controller_node");
  std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle); 

    
    //TODO make work for tortuga node?
    std::unique_ptr<RamNode> node;
    node.reset(new controlNode(n, 0.5));
    
    while (ros::ok()){
      node->update();
      ros::Duration(0.1).sleep();
    }
    
}
