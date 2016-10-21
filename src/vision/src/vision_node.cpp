//sg: this is going to be the primary vision node for qubo (or future robots whatever)
#include "vision_node.h"

//sg: I'm copying the pointer to a node handle scheme we used in the vehicle layer here 
VisionNode::VisionNode(std::shared_ptr<ros::NodeHandle> n, int rate, std::string feed0, std::string feed1, std::string feedb){

    //take in the node handle
    this->n = n;
    
    //this is really kind of ugly but eh.. we just make sure that all the camera feeds were valid.

    //init the first VideoCapture object
    cap0 = cv::VideoCapture(feed0);
    
    //make sure we have something valid
    if(!cap0.isOpened()){           
        ROS_ERROR("couldn't open file/camera 0  %s\n now exiting" ,feed0.c_str());
        exit(0);
    }
    
    //init second VideoCapture object
    cap1 = cv::VideoCapture(feed1);

    //make sure that worked too
    if(!cap1.isOpened()){           
        ROS_ERROR("couldn't open file/camera 1 %s\n now exiting" ,feed1.c_str());
        exit(0);
    }
    
    //init second VideoCapture object
    capb = cv::VideoCapture(feedb);
    
    //make sure that worked too
    if(!capb.isOpened()){            // capture a frame 
        ROS_ERROR("couldn't open file/camera b %s\n now exiting", feedb.c_str());
        exit(0);
    }



    //Need to register all our services 
    ros::ServiceServer service = this->n->advertiseService("buoy_detect", this->buoy_detector);
    
    
}



VisionNode::~VisionNode(){

    //sg: may need to close the cameras here not sure..
    
}

void VisionNode::update(){
    cap0 >> img0;
    cap1 >> img1;
    capb >> imgb;
    ros::spinOnce();
    //std::cout << img;
}



//Past this point is a collection of services and actions that will be able to called from any other node
//=================================================================================================================
    

bool VisionNode::buoy_detector(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res){
    // code goes here
    ROS_ERROR("You called the service! nice!");
    return 0;
}


//There are the defintions for all of our actionlib actions, may be moved to it's own class not sure yet. 
//=================================================================================================================
void VisionNode::test_execute(const ram_msgs::VisionExampleGoalConstPtr& goal, Server*as){
    //    goal->test_feedback = 5;
    ROS_ERROR("You called the action well done!");
    as->setSucceeded();
}
