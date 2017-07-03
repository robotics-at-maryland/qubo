#include "buoy_tuner.h"

//TODO put these in the header. 
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>


#define MAX_AREA 9000
#define MAX_CIRCULARITY 1000
#define MAX_RATIO 1000
#define MAX_CONVEXITY 1000


using namespace cv;
using namespace std;

//Constructor
BuoyActionTuner::BuoyActionTuner(actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> *as, VideoCapture cap):
	m_cap(cap),  //video capture, this only shows up in the tuner version of this action so that we can rewind the video etc. 
	m_as(as)    //action server handle ,may decide later to keep this outside, pass back a vector or something
 
{
	
   	//initialize background subtractors, keeping both in for now
	//you can decide which version of the subtractor you want to run by commenting in one of these line
	m_pMOG = createBackgroundSubtractorMOG2(10000, 35, false);
	//m_pMOG = bgsegm::createBackgroundSubtractorMOG(1000,5,.7,0);


    m_slider_area = 0;
    m_slider_circularity = 0;
    m_slider_convexity = 0;
    m_slider_ratio = 0;

    //create GUI window for the keypoints
    namedWindow("keypoints");

    //create all the trackbars
    createTrackbar( "area", "keypoints", &m_slider_area, MAX_AREA, areaCallback); 
    createTrackbar( "circularity", "keypoints", &m_slider_circularity, MAX_CIRCULARITY, circCallback);
    createTrackbar( "convexity", "keypoints", &m_slider_convexity, MAX_CONVEXITY, convCallback);
    createTrackbar( "inertia ratio", "keypoints", &m_slider_ratio, MAX_RATIO, inertiaCallback);

	

	// Change thresholds
	m_params.minThreshold = 0;
	m_params.maxThreshold = 256;
	//Filter by Area
	m_params.filterByArea = true;
	m_params.minArea = 100;
	// Set up detector with params
	m_detector = SimpleBlobDetector::create(m_params);


}



BuoyActionTuner::~BuoyActionTuner(){}
	

//TODO may want to bot pass the updateAction a Mat, let it edit the cap object
void BuoyActionTuner::updateAction(const Mat cframe) {   
	//create Background Subtractor objects
    
    Mat mog_output; //output from our background subtractor, we need to keep track of the unmodified current frame 
	vector<KeyPoint> keypoints; // Storage for blobs

	
	ram_msgs::VisionExampleFeedback feedback;
	Point2f center; 

	mog_output = backgroundSubtract(cframe); //updates the MOG frame
	
    m_detector->detect(mog_output, keypoints);
	imshow("Gray image" , mog_output);

	waitKey();

	ROS_ERROR("let's see if we see something");
	if (updateHistory(mog_output, keypoints, center)){
		feedback.x_offset = cframe.rows/2 - center.x; 
		feedback.y_offset = cframe.cols/2 - center.y;

		ROS_ERROR("publishing feedback");
		//I actually think it might be better to keep the action server away from this class, haven't decided yet..
		m_as->publishFeedback(feedback);
	}

}



//Apply's the MOG subtraction
Mat BuoyActionTuner::backgroundSubtract(const Mat cframe){

	Mat out_frame; // output matrix

	//update the background model
	m_pMOG->apply(cframe, out_frame);
	
    //blurs the image uses the MOG background subtraction
    GaussianBlur(out_frame, out_frame, Size(3,3), 0,0);

	
    // Define the structuring elements to be used in eroding and dilating the image 
    Mat se1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat se2 = getStructuringElement(MORPH_RECT, Size(5, 5));

    // Perform dialting and eroding helps to elminate background noise
	//sgillen@20175109-15:51 when I found this on the second operation was actually being performed
    morphologyEx(out_frame, out_frame, MORPH_CLOSE, se1);
    morphologyEx(out_frame, out_frame, MORPH_OPEN, se2);

    //inverts the colors 
    bitwise_not(out_frame, out_frame, noArray()); 


	return out_frame;
}



//Davids way of keeping track of the history of points
bool  BuoyActionTuner::updateHistory(const Mat cframe, vector<KeyPoint> keypoints, Point2f center){
    float pointX, pointY, x, y; 
    bool insert; 
    Vec3b color;
    int age = 10, offSet = 20, filter = 30, offSet2 = 5;//how long ago the blob was first seen and the offset of the center and the value for the color we want to see 

    //for every deteced blob either add it if its new or update current one 
    for (auto& point:keypoints ){    
        color = cframe.at<Vec3b>(point.pt); 
        cout << color << endl;
        insert = false;
        pointX = point.pt.x;
        pointY = point.pt.y;
        for (std::vector<tuple< Point2f, Vec3b, int >>::iterator it = m_history.begin(); it != m_history.end(); it++){ 
            x = std::get<0>(*it).x;
            y = std::get<0>(*it).y;   

            //if blob is within offSet pixels of a know blob update the blob to the new blobs location       
            if (((pointX <= x + offSet) && (pointX >= x - offSet)) && ((pointY <= y + offSet && (pointY >= y - offSet)))){
                m_history.erase(it);
                m_history.emplace_back(std::make_tuple (point.pt,color,0));
                insert = true;
            }
            std::get<2>(*it) += 1;
            //if the blobs hasnt been updated in age frames remove it 
            if (std::get<2>(*it) > age){
                m_history.erase(it);
            }
        }
        if (!insert)
            m_history.emplace_back(std::make_tuple (point.pt, color, 0));
    }
    //outputs the buoys offset if it is the right color.  
    for (std::vector<tuple< Point2f, Vec3b, int >>::iterator it = m_history.begin(); it != m_history.end(); it++){
        color = std::get<1>(*it);
        if (color[0] >= filter - offSet2 && color[0] <= filter + offSet2){
            center =  std::get<0>(*it);
            return true;}
    } 

    return false;
}



//------------------------------------------------------------------------------
//track bar functions

void BuoyActionTuner::areaCallback( int, void* )
{
    //Filter by Area.
    m_params.filterByArea = true;
    m_params.minArea = m_slider_area;
    m_params.maxArea = MAX_AREA;
}

void  BuoyActionTuner::circCallback( int, void* )
{
    //Filter by Circularity.
    m_params.filterByCircularity = true; 
    m_params.minCircularity = (float) m_slider_circularity / (float) MAX_CIRCULARITY;
}

void  BuoyActionTuner::convCallback( int, void* )
{
    //Filter by Convexity.
    m_params.filterByConvexity = true; 
    m_params.minConvexity = (float) m_slider_convexity / (float) MAX_CONVEXITY;
}

void  BuoyActionTuner::inertiaCallback( int, void* )
{
    //Filter by ratio of the inertia.
    m_params.filterByInertia = true; 
    m_params.minInertiaRatio = (float) m_slider_ratio / (float) MAX_RATIO;
}




