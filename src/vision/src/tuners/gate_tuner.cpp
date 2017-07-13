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

int kernel_size = 1; //kernel for the guassian blur
int kernel_max = 256;

VideoCapture cap;   

void thresh_callback(int, void* )
{

	cap.set(CV_CAP_PROP_POS_FRAMES, frame_num);

}

void blur_callback(int, void* )
{

	//the kernel for a guassian filter needs to be odd
	kernel_size = (round(kernel_size / 2.0) * 2) -1; //round down to nearest odd integer

	//make sure we don't have a negative number (error from round) or zero
	if (kernel_size < 1){
		kernel_size = 1;
	}

	//let the user know what the actual kernel being used is (kernel of one == no blur)
	setTrackbarPos("Kernel size","parameters", kernel_size);
	
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
	namedWindow("current frame");
	moveWindow("current frame", 20, 20);
	
	namedWindow("after blur");
	moveWindow("after blur", 220, 20);
	
	namedWindow("parameters");
	moveWindow("parameters", 420, 20);
		
	createTrackbar( "Canny thresh:", "parameters", &canny_thresh, max_thresh, thresh_callback );
	createTrackbar( "Hough thresh:", "parameters", &hough_thresh, max_thresh, thresh_callback );
	createTrackbar( "Kernel size", "parameters", &kernel_size, kernel_max, blur_callback);
	createTrackbar( "Frame", "parameters", &frame_num, max_frame, thresh_callback);
	
	thresh_callback( 0, 0 );

	
	Mat cframe; 

	while(true){	

		cap >> cframe;

		setTrackbarPos("Frame","parameters", cap.get(CV_CAP_PROP_POS_FRAMES));

		//redundant matrices so that we can display intermediate steps at the end
		Mat dst, cdst, gdst;


		GaussianBlur(cframe, gdst, Size( kernel_size, kernel_size ), 0, 0 );

		
		Canny(gdst, dst, canny_thresh, canny_thresh*ratio, 3);
		
		cvtColor(dst, cdst, CV_GRAY2BGR);
		vector<Vec4i> lines;
		vector<Vec2f> also_lines;
		HoughLinesP(dst, lines, 1, CV_PI/180, hough_thresh, 50, 10 );

		
		HoughLines(dst, also_lines, 1, CV_PI/180, hough_thresh, 50, 10 );
		
		for( size_t i = 0; i < also_lines.size(); i++ )
			{
				float rho = also_lines[i][0], theta = also_lines[i][1];
				//printf("line[%lu] = %f, %f \n", i, also_lines[i][0], also_lines[i][1]);
				Point pt1, pt2;
				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;
				pt1.x = cvRound(x0 + 1000*(-b));
				pt1.y = cvRound(y0 + 1000*(a));
				pt2.x = cvRound(x0 - 1000*(-b));
				pt2.y = cvRound(y0 - 1000*(a));
				line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
			}
		
		
		// for( size_t i = 0; i < lines.size(); i++ )
		// 	{
		// 		Vec4i l = lines[i];
		// 		line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
		// 		printf("line[%lu] = %lu, %lu \n", i, lines[i][0], lines[i][1]);
				
		// 	}
		
		
		//imshow("source", cframe);

		imshow("current frame" ,cframe);
		imshow("after blur", gdst);
		imshow("parameters", cdst);
		
		
		waitKey();
	}
	
}


