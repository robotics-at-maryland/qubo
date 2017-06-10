#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, char*argv[]){
    
    if(argc != 2){
        cout << "wrong number of arguments dummy, call \"a.out test_video\"" << endl;
        return 0;
    }

    
    
    // Read image
    VideoCapture cap = VideoCapture( argv[1] );
    Mat im;
    cap >> im;
    
    Mat im_with_keypoints;

    
    Mat planes[3];
    split(im, planes);
    imshow("plane1" , planes[0]);
    imshow("plane2" , planes[1]);
    imshow("plane3" , planes[2]);
    waitKey(100000);
    
    //const TermCriteria termCrit(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON);
    //EM test(1, (int)EM::COV_MAT_DIAGONAL, termCrit);

    if(im.empty()){
        cout << "something went wrong the image is empty! (maybe a bad path)" << endl;
        exit(0);
    }
    
    Ptr<EM> em = EM::create();

    vector<Mat> samples = {im};
    
    //em->trainEM(samples, im_with_keypoints, noArray(), noArray());
    
    
    em->predict2(im, im_with_keypoints);

    
    cout << im_with_keypoints;
    
    // //CvEMParams params(DEFAULT_NCLUSTERS, EM::COV_MAT_DIAGONAL, const termCrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, 1e-6))
        
    while(!im.empty()){
        waitKey(0);
        cap >> im;
    }
}
