// Simulated Camera
// pulls from the uwsim camera topic
// publishes a message of type sensor_msg/Image

#ifndef CAMERA_SIM_HEADER
#define CAMERA_SIM_HEADER

#include "ram_node.h"
#include "sensor_msgs/Image.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class CameraSimNode : public RamNode {
		
	public:
	CameraSimNode(std::shared_ptr<ros::NodeHandle>, int);
	~CameraSimNode();
	
	void update();
	void cameraCallBack(const sensor_msgs::Image msg);
	ros::NodeHandle getNode(){ return *n; };
	
	
	protected:
	sensor_msgs::Image msg;
	ros::Subscriber subscriber;
	
	/* this is a class used to convert the camera images into something
	 * opencv can understand.
	 */
	class ImageConverter{
		
		//These are used to allow us to subscribe to a compressed image,
		//instead of a raw
		image_transport::ImageTransport image_tran;
		image_transport::Subscriber image_sub;
		image_transport::Publisher image_pub;
		
		public:
		// Allows the converter to publish from the camera
		ImageConverter(const CameraSimNode& c_node);
		~ImageConverter();
		void imageCallBack(const sensor_msgs::ImageConstPtr& msg);
	};
	
	//CameraSimNode::ImageConverter image_con;

	
};

#endif
