#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat im;
Mat gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);


int main(int argc, char*argv[]){
    
    if(argc != 2){
        cout << "wrong number of arguments dummy, call \"a.out test_video\"" << endl;
        return 0;
    }
    

    // Read image
    VideoCapture cap = VideoCapture( argv[1] );
    cap >> im;
 

    // Convert input image to HSV

    Mat im_with_keypoints;
    std::vector<KeyPoint> keypoints;
  
    while(!im.empty()){


        
        cvtColor(im, gray, CV_BGR2GRAY);
        GaussianBlur(gray, gray, Size(15, 15), 2, 2 );
        
        /// Create Window
        char* source_window = "Source";
        namedWindow( source_window, CV_WINDOW_AUTOSIZE );
        imshow(source_window, im );


        Mat canny_output;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        int thresh = 84; //adjust this!!
        int max_thres = 255; //might want to play with this?
        
        /// Detect edges using canny
        Canny(gray, canny_output, thresh, thresh*2, 3 );
        
        namedWindow( "Canny", CV_WINDOW_AUTOSIZE );
        imshow( "Canny", canny_output );
        /// Find contours
        findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        
        /// Draw contours
        Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
            {
                
                Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
            }
        
        /// Show in a window
        namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        imshow( "Contours", drawing );
        
        
        
        waitKey(0);
        cap >> im;
    }

}
