#include <opencv2/opencv.hpp>

#include <iostream>


using namespace cv;
using namespace std;



int ratio = 3; //per canny's suggestion
int canny_thresh = 12; //starts at 12, this is what we will be changing though
int hough_thresh = 27;
int angle_tracker = 20;
int max_thresh = 255;//max for both thresh variable

double angle_thresh = .14;

int frame_num = 0; //keeps track of the current frame
int max_frame = 0; //total frames in the video. this may fail for cameras?

int kernel_size = 5; //kernel for the guassian blur
int kernel_max = 256;

int num_bins = 30; // needs to divide image width cleanly (not really though)
int max_bins = 100;


VideoCapture cap;


//all the thresh variables are already assigned without us needing to do anything here, so the only thing we need to do is set the frame_num if it was changed
//the trackbars only do ints, so we need to calculate a ratio for the angle threshold
void threshCallback(int, void* )
{

	angle_thresh = ((float) angle_tracker/ (float) max_thresh)*3.1415;
	cap.set(CV_CAP_PROP_POS_FRAMES, frame_num);

}

void blurCallback(int, void* )
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

	createTrackbar( "Canny thresh", "parameters", &canny_thresh, max_thresh, threshCallback );
	createTrackbar( "Hough thresh", "parameters", &hough_thresh, max_thresh, threshCallback );
	createTrackbar( "Angle thresh", "parameters", &angle_tracker, max_thresh, threshCallback );
	createTrackbar( "Num bins", "parameters", &num_bins, max_bins, threshCallback );
	createTrackbar( "Kernel size", "parameters", &kernel_size, kernel_max, blurCallback);
	createTrackbar( "Frame", "parameters", &frame_num, max_frame, threshCallback);

	threshCallback( 0, 0 );


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


		vector<int> xbin_count; //TODO better name
		for(int i = 0; i < num_bins; i++){
			xbin_count.push_back(0);
		}
		// int bin_size = cap.get( CAP_PROP_FRAME_WIDTH )/num_bins; typo maybe?
		int bin_size = cap.get( CV_CAP_PROP_FRAME_WIDTH )/num_bins;

		cout << "bin size = " << bin_size << endl;

		for( size_t i = 0; i < also_lines.size();i++) {
				float rho = also_lines[i][0], theta = also_lines[i][1];

				if (theta > 3.14 - angle_thresh && theta < 3.14 + angle_thresh){
					//printf("line[%lu] = %f, %f \n", i, also_lines[i][0], also_lines[i][1]);
					Point pt1, pt2;
					double a = cos(theta), b = sin(theta);
					double x0 = a*rho, y0 = b*rho;


					cout << "x0 =  " << x0 << "  num bins = " << num_bins <<  " bin = " << (int) (x0/bin_size)+1 << endl;
					int bin = (int) x0/bin_size;
					if(bin > 0){

						xbin_count[(int) ((x0/bin_size))]++;

						pt1.x = cvRound(x0 + 1000*(-b));
						pt1.y = cvRound(y0 + 1000*(a));
						pt2.x = cvRound(x0 - 1000*(-b));
						pt2.y = cvRound(y0 - 1000*(a));

						line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);

					}

					else {


						pt1.x = cvRound(x0 + 1000*(-b));
						pt1.y = cvRound(y0 + 1000*(a));
						pt2.x = cvRound(x0 - 1000*(-b));
						pt2.y = cvRound(y0 - 1000*(a));

						line( cdst, pt1, pt2, Scalar(0,255,0), 3, CV_AA);
					}

				}
			}

		for(int i = 0; i < xbin_count.size(); i++){
			cout << "bin" << i << "=" << " " << xbin_count[i] << endl;
		}

		//ok now xbin_count is populated, let's find which bin has the most lines


		int max = 0;
		int max_i = 0;

		for( int i = 0; i < xbin_count.size(); i++){
			if (xbin_count[i] > max ){
				max = xbin_count[i];
				max_i = i;
			}
		}

		int max2 = 0;
		int max2_i = 0;


		//the two is arbitrary and there are probably better ways to go about this
		for( int i = 0; i < xbin_count.size(); i++){
			if (xbin_count[i] > max2 && ( i > (max_i + 2)  || i < (max_i - 2 ))){
				max2 = xbin_count[i];
				max2_i = i;
			}
		}

		cout << "max1 - " << max_i << endl;
		cout << "max2 - " << max2_i << endl;

		//great lets find the average of our two location

		int average = ((bin_size*max_i + bin_size/2) + (bin_size*max2_i + bin_size/2))/2;

		Point pt1, pt2;
		pt1.x = (average);
		pt1.y = (1000);
		pt2.x = (average);
		pt2.y = (-1000);

		line( cdst, pt1, pt2, Scalar(255,0,0), 3, CV_AA);





		// for( size_t i = 0; i < lines.size(); i++ )
		// 	{
		// 		Vec4i l = lines[i];

		// 		printf("(%i, %i) (%i, %i) \n", l[0], l[1], l[2], l[3]);
		// 		double theta = atan2((l[0] - l[2]), (l[1] - l[3]));

		// 		cout << "theta" << theta  << endl;


		// 		// range is +- pi
		// 		if ( (abs(theta) < angle_thresh && abs(theta) > -angle_thresh) || (abs(theta) < (3.14 + angle_thresh)  && abs(theta)) > 3.14 - angle_thresh){
		// 			line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
		// 		 }

		// 	}


		//imshow("source", cframe);

		imshow("current frame" ,cframe);
		imshow("after blur", gdst);
		imshow("parameters", cdst);


		waitKey();
	}

}
