//sg: this is going to be the primary vision node for qubo (or future robots, whatever)
#include "vision_node.h"
using namespace std;


//you need to pass in a node handle, and a camera feed, which should be a file path either to a physical device or to a video  
VisionNode::VisionNode(std::shared_ptr<ros::NodeHandle> n, std::string feed)
    //initialize your server here, it's sort of a mess
    :example_server(*n, "vision_example", boost::bind(&VisionNode::find_buoy, this,  _1 , &example_server), false)
{
    //take in the node handle
    this->n = n;
    
    //init the first VideoCapture object
    cap = cv::VideoCapture(feed);
    
    //make sure we have something valid
    if(!cap.isOpened()){           
        ROS_ERROR("couldn't open file/camera  %s\n now exiting" ,feed.c_str());
        exit(0);
    }
    
    //register all services here
    //=====================================================================
    test_srv = this->n->advertiseService("service_test", &VisionNode::service_test, this);
    //buoy_detect_srv = this->n->advertiseService("buoy_detect", &VisionNode::buoy_detector, this);


    //start your action servers here
    //=====================================================================
	//	locate_buoy_act = example_server.start();
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

/*
* Past this point is a collection of services and 
* actions that will be able to called from any other node
* =================================================================================================================
*/
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

//if a buoy is found on frame finds where it is and returns the center offset 
void VisionNode::find_buoy(const ram_msgs::VisionExampleGoalConstPtr& goal, Server*as){
    float* center = processVideo(this->cap);
    cout << center << endl;
    as->setSucceeded();   
}

