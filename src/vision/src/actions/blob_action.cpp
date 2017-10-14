#include "blob_action.h"

using namespace cv;
using namespace std;


BlobAction::BlobAction(){

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;
	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 256;
	//Filter by Area
	params.filterByArea = true;
	params.minArea = 100;
	// Set up detector with params
	m_detector = SimpleBlobDetector::create(params);

}

BlobAction::~BlobAction(){}


//might want to return a point evnetually
int BlobAction::updateAction(const Mat cframe) {
	//create Background Subtractor objects
	vector<KeyPoint> keypoints; // Storage for blobs


	ram_msgs::VisionNavFeedback feedback;
	Point2f center;
	center.x = 0;
	center.y = 0;


	m_detector->detect(cframe, keypoints);

	int max = 0;
	for(int i = 0; i < keypoints.size(); i++){
		if(keypoints[i].size > max){
			max = keypoints[i].size;
			center = keypoints[i].pt;
		}
	}

	if(center.x){
		feedback.x_offset = cframe.rows/2 - center.x;
		feedback.y_offset = cframe.cols/2 - center.y;
	}
	else{ //this is a hack
		feedback.x_offset = 0;
	}

	return center.x;

}
