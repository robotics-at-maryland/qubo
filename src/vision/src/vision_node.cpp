//sg: this is going to be the primary vision node for qubo (or future robots, whatever)
#include "vision_node.h"

using namespace std;
using namespace ros;
using namespace AVT::VmbAPI;


int VisionNode::vmb_err(const int func_call, const string err_msg) {
	if (func_call != VmbErrorSuccess){
		ROS_ERROR("%s, code: %i", err_msg.c_str(), func_call);
		return func_call;
	}
	return func_call;
}


//you need to pass in a node handle, and a camera feed, which should be a file path either to a physical device or to a video  
VisionNode::VisionNode(NodeHandle n, NodeHandle np, string feed)
	:	m_buoy_server(n, "buoy_action", boost::bind(&VisionNode::findBuoy, this, _1, &m_buoy_server), false),
		m_gate_server(n, "gate_action", boost::bind(&VisionNode::findGate, this, _1, &m_gate_server), false),
		m_blob_server(n, "blob_action", boost::bind(&VisionNode::findBlob, this, _1, &m_blob_server), false)

{

	// isdigit makes sure checks if we're dealing with a number (like if want to open the default camera by passing a 0). If we are we convert our string to an int (VideoCapture won't correctly open the camera with the string in this case);
	//this could give us problems if we pass something like "0followed_by_string" but just don't do that.

	//sgillen@20170429-07:04 we really don't even need this... we can just pass in /dev/video0 if we want the web camera... I'll leave it for now though
	if(isdigit(feed.c_str()[0])){
		m_cap = cv::VideoCapture(atoi(feed.c_str()));
	}
	else{
		m_cap = cv::VideoCapture(feed);
	}


		//make sure we have something valid
    if(!m_cap.isOpened()){           
        ROS_ERROR("couldn't open file/camera  %s\n now exiting" ,feed.c_str());
        exit(0);
    }
	

	//TODO give option to user to specify the name of the video
	//TODO make sure this doesn't fail when specifying a directory that does not yet exist
	
	string output_dir;
	np.param<string>("output_dir", output_dir, ""); //this line will populate the output_dir variable if it's specified in the launch file

	//TODO change variable names
	if(!output_dir.empty()){
		
		stringstream output_ss;
		auto t = time(nullptr);
		auto tm = *localtime(&t);

		output_ss << output_dir;
		output_ss << put_time(&tm, "%Y%m%d_%H-%M-%S");
		output_ss << ".avi";
		
		string output_str = output_ss.str();
		
		int ex = static_cast<int>(m_cap.get(CV_CAP_PROP_FOURCC));

		cv::Size S = cv::Size((int) m_cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
					  (int) m_cap.get(CV_CAP_PROP_FRAME_HEIGHT));
		

		//sgillen@20172107-06:21 I found more problems trying to keep the extension (by passing ex as the second argument) than I did by forcing the output to be CV_FOURCC('M','J','P','G')  
		//m_output_video.open(output_str, ex, m_cap.get(CV_CAP_PROP_FPS), S, true);
		m_output_video.open(output_str, CV_FOURCC('M','J','P','G'), m_cap.get(CV_CAP_PROP_FPS), S, true);


		if(!m_output_video.isOpened()){
			ROS_ERROR("problem opening output video! we tried saving it as %s, now exiting" ,output_str.c_str());
			exit(0);
		}

		ROS_INFO("output video being saved as %s" , output_str.c_str());
	}

	
	//register all services here
	//------------------------------------------------------------------------------
	m_test_srv = n.advertiseService("service_test", &VisionNode::serviceTest, this);
	
	//start your action servers here
	//------------------------------------------------------------------------------
	m_buoy_server.start();
	m_gate_server.start();
	m_blob_server.start();
	ROS_INFO("servers started");
}


VisionNode::~VisionNode(){
    //sg: may need to close the cameras here not sure..
}

void VisionNode::update(){

	m_cap >> m_img;
	//if one of our frames was empty it means we ran out of footage, should only happen with test feeds or if a camera breaks I guess
	if(m_img.empty()){           
		ROS_ERROR("ran out of video (one of the frames was empty) exiting node now");
		exit(0);
	}
	
	//if the user didn't specify a directory this will not be open
	if(m_output_video.isOpened()){
		ROS_ERROR("writing image!");
		m_output_video << m_img;
	}
	
	spinOnce();
}


/*
* Past this point is a collection of services and 
* actions that will be able to called from any other node
* =================================================================================================================
*/
bool VisionNode::serviceTest(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res){
    ROS_ERROR("service called successfully");
}



//There are the definitions for all of our actionlib actions, may be moved to it's own class not sure yet. 
//=================================================================================================================
void VisionNode::testExecute(const ram_msgs::VisionNavGoalConstPtr& goal, actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *as){
    //    goal->test_feedback = 5;
    ROS_ERROR("You called the action well done!");
    as->setSucceeded();
}


//if a buoy is found on frame finds where it is and returns the center offset 
void VisionNode::findBuoy(const ram_msgs::VisionNavGoalConstPtr& goal,  actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *as){
	
	BuoyAction action = BuoyAction(as);
	
	while(true){
		action.updateAction(m_img); //this will also publish the feedback
	}
	
	as->setSucceeded();   
}

void VisionNode::findGate(const ram_msgs::VisionNavGoalConstPtr& goal,  actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *as){
	
	GateAction action = GateAction();

	while(true){
		ROS_ERROR("updating action");
		action.updateAction(m_img);
	}

	as->setSucceeded();
}


void VisionNode::findBlob(const ram_msgs::VisionNavGoalConstPtr& goal,  actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *as){
	
	BlobAction action = BlobAction(as);

	while(true){
		ROS_ERROR("updating action");
		action.updateAction(m_img);
	}

	as->setSucceeded();
}
