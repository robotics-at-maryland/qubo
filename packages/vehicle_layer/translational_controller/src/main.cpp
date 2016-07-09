#include "translational_controller.h"

int main(int argc, char **argv) {
    std::shared_ptr<ros::NodeHandle> n; 
    ros::init(argc, argv, "translational_controller");
    
    //TODO make work for tortuga node?
    std::unique_ptr<RamNode> node;
    node.reset(new TranslationalController(n, 10));
    
    while (ros::ok()){
        node->update();
    }
    
}
