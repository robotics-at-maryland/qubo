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
		m_gate_server(n, "gate_action", boost::bind(&VisionNode::findGate, this, _1, &m_gate_server), false)
{

	// isdigit makes sure checks if we're dealing with a number (like if want to open the default camera by passing a 0). If we are we convert our string to an int (VideoCapture won't correctly open the camera with the string in this case);
	//this could give us problems if we pass something like "0followed_by_string" but just don't do that.

	//sgillen@20170429-07:04 we really don't even need this... we can just pass in /dev/video0 if we want the web camera... I'll leave it for now though
	if(feed == "IP") {
		// use the mako if feed isn't given
		//start the camera

		ROS_ERROR("getting Vimba");
		auto& m_vimba_sys = VimbaSystem::GetInstance();
		if( vmb_err(m_vimba_sys.Startup(), "Unable to start the Vimba System") ) { exit(1); }

		ROS_ERROR("finding cameras");
		CameraPtrVector c_vec;
		if( vmb_err(m_vimba_sys.GetCameras( c_vec ), "Unable to find cameras") ) {exit(1); }
		// Since we only have one camera currently, we can just use the first thing returned when getting all the cameras
		m_gige_camera = c_vec[0];
		string camera_id;
		if (vmb_err(m_gige_camera->GetID(camera_id), "Unable to get a camera ID from Vimba") ){ exit(1); }
		while( VmbErrorSuccess != vmb_err(m_gige_camera->Open(VmbAccessModeFull), "Error opening a camera with Vimba") ) {}

		// We need to height, width and pixel format of the camera to convert the images to something OpenCV likes
		FeaturePtr feat;
		if(!vmb_err(m_gige_camera->GetFeatureByName("GVSPAdjustPacketSize", feat), "Error getting packet feature")){
			if(!vmb_err(feat->RunCommand(), "Error running packet command")){
				bool done = false;
				while (!done) {
					if( vmb_err(feat->IsCommandDone(done), "Error getting command status") ) { break; }
				}
			}
		}
		if(!vmb_err(m_gige_camera->GetFeatureByName( "Width", feat), ("Error getting the camera width" ))){
			VmbInt64_t width;
			if (!vmb_err(feat->GetValue(width), "Error getting width")) {
				ROS_ERROR("Width: %lld", width);
				m_width = width;
			}
		}
		if(!vmb_err(m_gige_camera->GetFeatureByName( "Height", feat), ("Error getting the camera height" ))){
			VmbInt64_t height;
			if (!vmb_err(feat->GetValue(height), "Error getting height")) {
				ROS_ERROR("Height: %lld", height);
				m_height = height;
			}
		}
		if(!vmb_err(m_gige_camera->GetFeatureByName("PixelFormat", feat), "Error getting pixel format")){
			if( vmb_err(feat->SetValue(VmbPixelFormatBayerBG8), "Error setting pixel format to bgr8") ){
				vmb_err(feat->SetValue(VmbPixelFormatMono8), "Error setting pixel format to mono8");
			}
			vmb_err(feat->GetValue(m_pixel_format), "Error getting format");
		}
		m_observer = IFrameObserverPtr(new FrameObserver(m_gige_camera));
		vmb_err(m_gige_camera->StartContinuousImageAcquisition( 3, IFrameObserverPtr(m_observer)), "Error starting continuous image acquisition");

		m_img = cv::Mat (m_height, m_width, CV_8UC3);
	} else {

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
		m_width = m_cap.get(CV_CAP_PROP_FRAME_WIDTH);
		m_height = m_cap.get(CV_CAP_PROP_FRAME_HEIGHT);
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

		cv::Size S = cv::Size((int) m_width,    // Acquire input size
							  (int) m_height);


		//sgillen@20172107-06:21 I found more problems trying to keep the extension (by passing ex as the second argument) than I did by forcing the output to be CV_FOURCC('M','J','P','G')
		//m_output_video.open(output_str, ex, m_cap.get(CV_CAP_PROP_FPS), S, true);
		m_output_video.open(output_str, CV_FOURCC('M','J','P','G'), 20/* m_cap.get(CV_CAP_PROP_FPS) */, S, true);


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
	ROS_INFO("servers started");

}


VisionNode::~VisionNode(){
	//sg: may need to close the cameras here not sure..
	auto& m_vimba_sys = VimbaSystem::GetInstance();
	m_gige_camera->StopContinuousImageAcquisition();
	m_gige_camera->Close();
	m_vimba_sys.Shutdown();
}

void VisionNode::update(){

	// Use the mako if its present
	if (m_gige_camera != nullptr){
		ROS_ERROR("Get vimba frame: %i", m_img.data);
		getVmbFrame(m_img);
		// ROS_ERROR("%s", m_img.data);
	} else {
		m_cap >> m_img;
	}
	//if one of our frames was empty it means we ran out of footage, should only happen with test feeds or if a camera breaks I guess
	if(m_img.empty()){
		// ROS_ERROR("ran out of video (one of the frames was empty) exiting node now");
		// exit(0);
		ROS_ERROR("Empty Frame");
		return;
	}

	//if the user didn't specify a directory this will not be open
	if(m_output_video.isOpened()){
		ROS_ERROR("writing image!");
		m_output_video << m_img;
	}

	spinOnce();
}

VisionNode::FrameObserver::FrameObserver(CameraPtr& camera) : IFrameObserver(camera) {
	ROS_ERROR("observer create");
}

void VisionNode::FrameObserver::FrameReceived( const FramePtr frame) {
	VmbFrameStatusType status;
	ROS_ERROR("Frame rec");
	vmb_err(frame->GetReceiveStatus(status), "Error getting frame status");
	if(status != VmbFrameStatusComplete){
		ROS_ERROR("Bad frame received, code: %i", status);
	}
	m_q_frame.push(frame);
}

FramePtr VisionNode::FrameObserver::GetFrame() {
	FramePtr f;
	if(m_q_frame.empty()) {
		return f;
	}
	f = m_q_frame.front();
	m_q_frame.pop();
	return f;
}

void VisionNode::getVmbFrame(cv::Mat& cv_frame){
	if(m_gige_camera == nullptr) {
		ROS_ERROR("Attempted to get a frame from a null Vimba camera pointer");
		return;
	}

	FramePtr vmb_frame = SP_DYN_CAST(m_observer, FrameObserver)->GetFrame();
	if(vmb_frame == nullptr) {
		ROS_ERROR("Null frame");
		return;
	}
	// if( vmb_err( m_gige_camera->AcquireMultipleImages(vmb_frame, 5000), "Error getting single frame" ) ) { return; }
	// VmbFrameStatusType status;
	// vmb_frame->GetReceiveStatus(status);
	// if(status != VmbFrameStatusComplete) {
	// 	ROS_ERROR("Malformed frame received, code %i", status);
	// }
	VmbUint32_t size;
	vmb_frame->GetImageSize(size);
	ROS_ERROR("size");
	unsigned char* img_buf;
	if( vmb_err( vmb_frame->GetImage(img_buf), "Error getting image buffer")) { return; }

	VmbImage img_src, img_dest;
	img_src.Size = sizeof( img_src );
	img_dest.Size = sizeof( img_dest );

	ROS_ERROR("VmbImage");
	vmb_err( VmbSetImageInfoFromPixelFormat( m_pixel_format, m_width, m_height, &img_src) , "error px format 1");
	vmb_err( VmbSetImageInfoFromPixelFormat(VmbPixelFormatBgr8, m_width, m_height, &img_dest) , "error px format 2");

	img_src.Data = img_buf;
	img_dest.Data = cv_frame.data;

	ROS_ERROR("%i, %i", img_buf, cv_frame.data);
	ROS_ERROR("Transform");
	vmb_err( VmbImageTransform(&img_src, &img_dest, NULL, 0), "Error transforming image" );
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
