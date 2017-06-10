//sg: this is going to be the primary vision node for qubo (or future robots, whatever)
#include "vision_node.h"

using namespace std;


//you need to pass in a node handle, and a camera feed, which should be a file path either to a physical device or to a video  
VisionNode::VisionNode(ros::NodeHandle n, std::string feed_topic)
  
	:m_it(n),
	 buoy_server(n, "buoy_action", boost::bind(&VisionNode::findBuoy, this,  _1 , &buoy_server), false)
{
    //take in the node handle
    this->n = n;

	//TODO resolve namespaces pass in args etc
	m_image_sub =  m_it.subscribe("qubo/camera/image_raw", 1 , &VisionNode::imageCallback, this);


	//register all services here
    //------------------------------------------------------------------------------
    test_srv = this->n.advertiseService("service_test", &VisionNode::serviceTest, this);
    buoy_detect_srv = this->n.advertiseService("buoy_detect", &VisionNode::buoyDetector, this);


    //start your action servers here
    //------------------------------------------------------------------------------
	buoy_server.start();
	ROS_ERROR("server started");
}


VisionNode::~VisionNode(){
    //sg: may need to close the cameras here not sure..
}

void VisionNode::update(){
	
	ros::spinOnce();
   
}

void VisionNode::imageCallback(const sensor_msgs::ImageConstPtr& msg){

	cv_bridge::CvImagePtr cv_ptr;
    try
		{
			//TODO can we do this without a copy?
			cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
			img = cv_ptr->image; //this could be bad if img does not copy anything, even if it does 
		}
    catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
	
}

/*
* Past this point is a collection of services and 
* actions that will be able to called from any other node
* =================================================================================================================
*/
bool VisionNode::serviceTest(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res){
    ROS_ERROR("service called successfully");
}

//this will detect if there are buoy's in the scene or not. 
bool VisionNode::buoyDetector(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res){
    
    //sg - copied this from stack overflow, you can call it but it exits with a (handled) exception somewhere

    // // Set up the detector with default parameters.
    // cv::SimpleBlobDetector detector;
 
    // // Detect blobs.
    std::vector<cv::KeyPoint> keypoints;
    // detector.detect(img, keypoints);

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(); 
	detector->detect( img, keypoints );
	
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
void VisionNode::testExecute(const ram_msgs::VisionExampleGoalConstPtr& goal, actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> *as){
    //    goal->test_feedback = 5;
    ROS_ERROR("You called the action well done!");
    as->setSucceeded();
}

//if a buoy is found on frame finds where it is and returns the center offset 
void VisionNode::findBuoy(const ram_msgs::VisionExampleGoalConstPtr& goal,  actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> *as){
	ROS_ERROR("here!");

	FindBuoyAction action = FindBuoyAction(as);
	
	while(true){
		action.updateAction(img); //this will also publish the feeback
	}
	
	as->setSucceeded();   
}

