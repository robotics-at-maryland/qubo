#include "rotational_controller.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "rotational_controller");
    std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);
    
    //TODO make work for tortuga node?
    std::unique_ptr<RamNode> node;
    node.reset(new RotationalController(n, 10));
   
    while (ros::ok()){
        node->update();
    }
   
    return 0; 
}
