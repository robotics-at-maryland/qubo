#include "vision_node.h"

//TODO find some standardized way to keep videos in our repo, maybe just commit them? does github have a size limit?
#define TEST_FEED "/home/sgillen/not_bouy.avi"


int main(int argc, char* argv[]){

    ros::init(argc,argv, "vision_node");
    std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);

    VisionNode node(n,10,TEST_FEED);

    ros::ServiceServer service = n->advertiseService("buoy_detect", VisionNode::buoy_detector);

    while(ros::ok()){
        node.update();
    }
    
    return 0;
}
