#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

CascadeClassifier buoy_cascade;

String window_name = "Buoy Detection";
int main(int argc, char *argv[]){
    VideoCapture capture;
    Mat frame;
    //-- 1. Load the cascades
    if( !buoy_cascade.load( argv[1] ) ){ printf("--(!)Error loading buoy cascade\n"); return -1; };
    //-- 2. Read the video stream
    capture.open(argv[2]);
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        char c = (char)waitKey(10);
        if( c == 27 ) { break; } // escape
    }
    return 0;
}

void detectAndDisplay( Mat frame ){
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    //TODO: Find out what parameters mean?
    buoy_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(64, 48) );
    for ( size_t i = 0; i < faces.size(); i++ ){
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        //cout << i << " " << faces[i].x << " " << faces[i].y << endl;

        Mat f = frame.clone();
        Mat faceROI = frame_gray( faces[i] );
        Scalar average = mean( f, faceROI );
	//cout << faceROI.dims << " " << faceROI.rows << " " << faceROI.cols <<  endl;
        cout << i << " " << average[0] << " " << average[1] << " "  <<  average[2] << endl;
    }
    //-- Show what you got
    imshow( window_name, frame );
}
