//sg: this is going to be the primary vision node for qubo (or future robots, whatever)
#include "vision_node.h"

//you need to pass in a node handle, a rate to poll at, and 3 camera feeds, which should be a file path either to a physical device or to a video file
//feed0 and feed1 need to correspond to the two forward facing cameras, feedb is the bottom facing camera. 
//note you always need to pass 3 feeds even if you're just testing monocualar tasks
VisionNode::VisionNode(std::shared_ptr<ros::NodeHandle> n, std::string feed){
    //take in the node handle
    this->n = n;
    
    //this is really kind of ugly but eh.. we just make sure that all the camera feeds were valid.
    
    //init the first VideoCapture object
    cap = cv::VideoCapture(feed);
    
    //make sure we have something valid
    if(!cap.isOpened()){           
        ROS_ERROR("couldn't open file/camera 0  %s\n now exiting" ,feed.c_str());
        exit(0);
    }
    
    //register all services here
    //=====================================================================
    test_srv = this->n->advertiseService("service_test", &VisionNode::service_test, this);
    buoy_detect_srv = this->n->advertiseService("buoy_detect", &VisionNode::buoy_detector, this);
    
}


VisionNode::~VisionNode(){
    //sg: may need to close the cameras here not sure..
}

void VisionNode::update(){
    cap >> img;
    //if one of our frames was empty it means we ran out of footage, should only happen with test feeds or if a camera breaks I guess
    if(img.empty()){           
        ROS_ERROR("ran out of video (one of the frames was empty) exiting node now");
        exit(0);
    }

    //check if anyone wanted a service or action started since the last time we looked
    ros::spinOnce();
}

//Past this point is a collection of services and actions that will be able to called from any other node
//=================================================================================================================
bool VisionNode::service_test(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res){
    ROS_ERROR("service called successfully");
}

//this will detect if there are buoy's in the scene or not. 
bool VisionNode::buoy_detector(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res){
    
    //sg - copied this from stack overflow, you can call it but it exits with a (handled) exception somewhere

    // Set up the detector with default parameters.
    cv::SimpleBlobDetector detector;
 
    // Detect blobs.
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(img, keypoints);
 
    // Draw detected blobs as red circles.
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
    cv::Mat im_with_keypoints;
    cv::drawKeypoints(img, keypoints, im_with_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
 
    // Show blobs
    cv::imshow("keypoints", im_with_keypoints );
    cv::waitKey(0);

}

//There are the definitions for all of our actionlib actions, may be moved to it's own class not sure yet. 
//=================================================================================================================
void VisionNode::test_execute(const ram_msgs::VisionExampleGoalConstPtr& goal, Server*as){
    //    goal->test_feedback = 5;
    ROS_ERROR("You called the action well done!");
    as->setSucceeded();
}
