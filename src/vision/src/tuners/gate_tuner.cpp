#include <opencv2/opencv.hpp>

#include <iostream>


using namespace cv;
using namespace std;



int ratio = 3; //per canny's suggestion
int canny_thresh = 100; //starts at 100, this is what we will be changing though 
int hough_thresh = 100;
int max_thresh = 255;//max for both thresh variable

int frame_num = 0; //keeps track of the current frame
int max_frame = 0; //total frames in the video. this may fail for cameras?

VideoCapture cap;   

void thresh_callback(int, void* )
{

	
	
	cap.set(CV_CAP_PROP_POS_FRAMES, frame_num);
	cout << "callback " << endl;
}


int main(int argc, char* argv[]){
	
	
	//check for the input parameter correctness
    if(argc != 2){
        cerr <<"Incorrect input list, usage: rosrun vision gate_tuner <path_to_video_or_camera>" << endl;
		exit(1);
	}

	//create and open the capture object

	cap.open(argv[1]);


	max_frame = cap.get(CV_CAP_PROP_FRAME_COUNT );
	cout << max_frame << endl; 
	
    
    if(!cap.isOpened()){
        //error in opening the video input
        cerr << "Unable to open video file: " << argv[1] << endl;
        exit(1);
    }



	//make some windows, place them at 20 pixels out because my window manager can't grab them in the corner..
	namedWindow("Canny output");
	moveWindow("Canny output", 20, 20);
	
	namedWindow("detected lines");
	moveWindow("detected lines", 20, 20);

	namedWindow("parameters");
	moveWindow("parameters", 20, 20);
		
	createTrackbar( "Canny thresh:", "parameters", &canny_thresh, max_thresh, thresh_callback );
	createTrackbar( "Hough thresh:", "parameters", &hough_thresh, max_thresh, thresh_callback );
	createTrackbar( "Frame", "parameters", &frame_num, max_frame, thresh_callback);
	thresh_callback( 0, 0 );

	
	
	Mat cframe; 

	while(true){	

		cap >> cframe;

		setTrackbarPos("Frame","parameters", cap.get(CV_CAP_PROP_POS_FRAMES));
		
		Mat dst, cdst;
		Canny(cframe, dst, canny_thresh, canny_thresh*ratio, 3);
		
		cvtColor(dst, cdst, CV_GRAY2BGR);
		vector<Vec4i> lines;
		HoughLinesP(dst, lines, 1, CV_PI/180, hough_thresh, 50, 10 );
		
		for( size_t i = 0; i < lines.size(); i++ )
			{
				Vec4i l = lines[i];
				line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
			}
		
		
		//imshow("source", cframe);
		imshow("Canny output", dst);
		imshow("parameters", cdst);
		
		
		waitKey();
	}
	
}


