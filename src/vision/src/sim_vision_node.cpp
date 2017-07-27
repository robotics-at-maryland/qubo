//sg: this is going to be the primary vision node for qubo (or future robots, whatever)
#include "sim_vision_node.h"

using namespace std;
using namespace ros;

//you need to pass in a node handle, and a camera feed, which should be a file path either to a physical device or to a video  
SimVisionNode::SimVisionNode(NodeHandle n, NodeHandle np, string feed_topic)
	:m_it(n), //image transport
	 m_buoy_server(n, "buoy_action", boost::bind(&SimVisionNode::findBuoy, this, _1, &m_buoy_server), false),
	 m_gate_server(n, "gate_action", boost::bind(&SimVisionNode::findGate, this, _1, &m_gate_server), false)
{
	
	//TODO resolve namespaces pass in args etc
	m_image_sub =  m_it.subscribe(feed_topic, 1 , &SimVisionNode::imageCallback, this);
	
	//register all services here
	//------------------------------------------------------------------------------
	m_test_srv = n.advertiseService("service_test", &SimVisionNode::serviceTest, this);
	
	//start your action servers here
	//------------------------------------------------------------------------------
	m_buoy_server.start();
	m_gate_server.start();
	ROS_INFO("servers started");
}


SimVisionNode::~SimVisionNode(){
    //sg: may need to close the cameras here not sure..
}

void SimVisionNode::update(){
	
	spinOnce();
}


//TODO, we can get even more performance gains if we set a marker telling us if an image is stale or not, if it is we can just save the last response to a service or whatever and return that value
void SimVisionNode::imageCallback(const sensor_msgs::ImageConstPtr& msg){

	cv_bridge::CvImagePtr cv_ptr;
	
    try	{
		//TODO can we do this without a copy?
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		m_img = cv_ptr->image; //this could be bad if img does not copy anything, even if it does
	}
    catch (cv_bridge::Exception& e){
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	
}

/*
* Past this point is a collection of services and 
* actions that will be able to called from any other node
* =================================================================================================================
*/
bool SimVisionNode::serviceTest(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res){
    ROS_ERROR("service called successfully");
}



//There are the definitions for all of our actionlib actions, may be moved to it's own class not sure yet. 
//=================================================================================================================
void SimVisionNode::testExecute(const ram_msgs::VisionNavGoalConstPtr& goal, actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *as){
    //    goal->test_feedback = 5;
    ROS_ERROR("You called the action well done!");
    as->setSucceeded();
}


//if a buoy is found on frame finds where it is and returns the center offset 
void SimVisionNode::findBuoy(const ram_msgs::VisionNavGoalConstPtr& goal,  actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *as){
	
	BuoyAction action = BuoyAction(as);
	
	while(true){
		action.updateAction(m_img); //this will also publish the feedback
	}
	
	as->setSucceeded();   
}

void SimVisionNode::findGate(const ram_msgs::VisionNavGoalConstPtr& goal,  actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *as){
	
	GateAction action = GateAction();

	while(true){
		ROS_ERROR("updating action");
		action.updateAction(m_img);
	}

	as->setSucceeded();
}
