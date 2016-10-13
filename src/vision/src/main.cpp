#include "vision_node.h"

//TODO find some standardized way to keep videos in our repo, maybe just commit them? does github have a size limit?
#define TEST_FEED "/home/sgillen/not_bouy.avi"
typedef actionlib::SimpleActionServer<ram_msgs::visionExampleAction> Server;




void execute(const ram_msgs::bool_bool_intGoalConstPtr& goal, Server*as){
    //goal->test_feedback = 5;
    ROS_ERROR("You called the action well done!");
    as->setSucceeded();
}


int main(int argc, char* argv[]){

    ros::init(argc,argv, "vision_node");
    std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);

    VisionNode node(n,10,TEST_FEED);

    //    ros::ServiceServer service = n->advertiseService("buoy_detect", VisionNode::buoy_detector);
    
    Server server(n, "bool_bool_int", boost::bind(&execute,_1, &server), false);
    server.start();


    while(ros::ok()){
        node.update();
    }
    
    return 0;
}
