#include <stdio.h>
#include <stdint.h>
#include <dc1394/dc1394.h>
#include <stdlib.h>
#include <inttypes.h>

#include <endian.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


#ifndef _WIN32
#include <unistd.h>
#endif


/*-----------------------------------------------------------------------
 *  Releases the cameras and exits
 *-----------------------------------------------------------------------*/
void cleanup_and_exit(dc1394camera_t *camera)
{
    dc1394_video_set_transmission(camera, DC1394_OFF);
    dc1394_capture_stop(camera);
    dc1394_camera_free(camera);
    exit(1);
}

int main(int argc, char *argv[])
{
    FILE* imagefile;
    dc1394camera_t *camera;
    unsigned int width, height;
    dc1394video_frame_t *frame=NULL;
    dc1394video_frame_t stereo;

    //dc1394featureset_t features;
    dc1394_t * d;
    dc1394camera_list_t * list;
    dc1394error_t err;     
    //
    // Create an OpenCV window
    //
    cv::namedWindow( "Stereo image", CV_WINDOW_AUTOSIZE );             

    d = dc1394_new ();
    if (!d)
        return 1;
    err=dc1394_camera_enumerate (d, &list);
    DC1394_ERR_RTN(err,"Failed to enumerate cameras");

    if (list->num == 0) {
        dc1394_log_error("No cameras found");
        return 1;
    }

    camera = dc1394_camera_new (d, list->ids[0].guid);
    if (!camera) {
        dc1394_log_error("Failed to initialize camera with guid %llx", list->ids[0].guid);
        return 1;
    }
    dc1394_camera_free_list (list);

    printf("Using camera with GUID %i \n", camera->guid);


    /*-----------------------------------------------------------------------
     *  setup capture
     *-----------------------------------------------------------------------*/

    err=dc1394_video_set_operation_mode(camera, DC1394_OPERATION_MODE_LEGACY );
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set 1394A mode");

    err=dc1394_video_set_iso_speed(camera, DC1394_ISO_SPEED_400 );
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set iso speed");

    err=dc1394_video_set_mode(camera, DC1394_VIDEO_MODE_FORMAT7_3 );
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set video mode\n");

    err=dc1394_format7_set_roi(camera, DC1394_VIDEO_MODE_FORMAT7_3, DC1394_COLOR_CODING_RAW16, DC1394_USE_MAX_AVAIL, 0, 0, 1024, 768 );       
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not ROI\n");

    err=dc1394_capture_setup(camera,4, DC1394_CAPTURE_FLAGS_DEFAULT );
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not setup camera-\nmake sure that the video mode and framerate are\nsupported by your camera\n");


    /*-----------------------------------------------------------------------
     *  GIJS edit: show F7 info
     *-----------------------------------------------------------------------*/
    uint32_t unit_bytes;
    uint32_t max_bytes;
    err=dc1394_format7_get_packet_parameters(camera, DC1394_VIDEO_MODE_FORMAT7_3, &unit_bytes, &max_bytes );
    printf("\n[DEBUG] F7 Info\n");
    printf("[DEBUG] unit_byte: %d (error %d)\n", unit_bytes, err );   
    printf("[DEBUG] max_bytes: %d (error %d)\n", max_bytes, err );
    uint32_t packet_size;
    err=dc1394_format7_get_packet_size(camera, DC1394_VIDEO_MODE_FORMAT7_3, &packet_size );
    printf("[DEBUG] packet_size: %d (error %d)\n", packet_size, err );
    uint32_t packets_per_frame;
    err=dc1394_format7_get_packets_per_frame(camera, DC1394_VIDEO_MODE_FORMAT7_3, &packets_per_frame );     
    printf("[DEBUG] packets_per_frame: %d (error %d)\n", packets_per_frame, err );
    uint32_t pixels_per_frame;
    err=dc1394_format7_get_pixel_number(camera, DC1394_VIDEO_MODE_FORMAT7_3, &pixels_per_frame );   
    printf("[DEBUG] pixels_per_frame: %d (error %d)\n", pixels_per_frame, err );
    uint32_t recommended_packet_size;
    err=dc1394_format7_get_recommended_packet_size(camera, DC1394_VIDEO_MODE_FORMAT7_3, &recommended_packet_size );
    printf("[DEBUG] recommended_packet_size: %d (error %d)\n", recommended_packet_size, err );
    uint64_t total_bytes;
    err=dc1394_format7_get_total_bytes(camera, DC1394_VIDEO_MODE_FORMAT7_3, &total_bytes );
    printf("[DEBUG] total_size: %d (error %d)\n", total_bytes, err );


    /*-----------------------------------------------------------------------
     *  have the camera start sending us data
     *-----------------------------------------------------------------------*/
    err=dc1394_video_set_transmission(camera, DC1394_ON);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not start camera iso transmission\n");


    //
    // Test loop
    //
    cv::Mat StereoImage = cv::Mat( 1536, 1024, CV_8UC1 );
    cv::Mat dispImage   = cv::Mat( 1536, 1024, CV_8UC3 );
    short int* source;
    short int* dest;

    for( int n = 0; n < 1000; n++ )
    {

      /*-----------------------------------------------------------------------
      *  capture one frame
      *-----------------------------------------------------------------------*/
      err=dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, &frame);
      DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not capture a frame\n");


      /*-----------------------------------------------------------------------
      *  check if frame is corrupt
      *-----------------------------------------------------------------------*/      
      if( dc1394_capture_is_frame_corrupt( camera, frame) )
      printf("\n[DEBUG] frame is corrupt!\n"); 

      /*-----------------------------------------------------------------------
      *  copy the buffer data, but only copy the pointer to the image
      *  we don't allocate the image buffer of tempframe (pointer copy from the DMA ring buffer)
      *  so the alloc should be zero
      *-----------------------------------------------------------------------*/ 
      memcpy( &stereo, frame, sizeof(dc1394video_frame_t) );
      stereo.allocated_image_bytes=0;


      /*-----------------------------------------------------------------------
      *  check if frame is corrupt
      *-----------------------------------------------------------------------*/ 
      stereo.image=NULL;
      dc1394_deinterlace_stereo_frames( frame, &stereo, DC1394_STEREO_METHOD_INTERLACED );    


      /*-----------------------------------------------------------------------
      *  copy to opencv
      *-----------------------------------------------------------------------*/          
      dest   = (short int*)&StereoImage.data[0];
      source = (short int*)&stereo.image[0];
      memcpy( dest, source, 1536*1024 );                


      /*-----------------------------------------------------------------------
     *  show frame info
     *-----------------------------------------------------------------------*/
      printf("\n[DEBUG] Frame Info\n");
      printf("[DEBUG] image_bytes: %d\n",           stereo.image_bytes);
      printf("[DEBUG] size[0]: %d\n",               stereo.size[0]);
      printf("[DEBUG] size[1]: %d\n",               stereo.size[1]);      
      printf("[DEBUG] allocated_image_bytes: %d\n", stereo.allocated_image_bytes );
      printf("[DEBUG] total_bytes: %d\n",           stereo.total_bytes );
      printf("[DEBUG] color_coding: %d\n",          stereo.color_coding);     
      printf("[DEBUG] color_filter: %d\n",          stereo.color_filter);
      printf("[DEBUG] packet_size: %d\n",           stereo.packet_size);      
      printf("[DEBUG] packets_per_frame: %d\n",     stereo.packets_per_frame);
      printf("[DEBUG] padding_bytes: %d\n",         stereo.padding_bytes);
      printf("[DEBUG] timestamp: %d\n",             stereo.timestamp);
      printf("[DEBUG] stride: %d\n",                stereo.stride);
      printf("[DEBUG] data_depth: %d\n",            stereo.data_depth);
      printf("[DEBUG] id: %d\n",                    stereo.id);
      printf("[DEBUG] frames_behind: %d\n",         stereo.frames_behind);
      printf("[DEBUG] image: %u\n",                 stereo.image); 


      /*-----------------------------------------------------------------------
      *  convert to color image
      *-----------------------------------------------------------------------*/
      cvtColor( StereoImage, dispImage, CV_BayerGR2RGB );


      /*-----------------------------------------------------------------------
      *  Show OpenCV image
      *-----------------------------------------------------------------------*/
      imshow("Stereo image", dispImage );
      cv::waitKey(10);


      /*-----------------------------------------------------------------------
      *  Give back frame to queue
      *-----------------------------------------------------------------------*/
      dc1394_capture_enqueue(camera,frame);


      /*-----------------------------------------------------------------------
      * Free the memory
      *-----------------------------------------------------------------------*/
      free(stereo.image);

    }

    /*-----------------------------------------------------------------------
     *  stop data transmission
     *-----------------------------------------------------------------------*/
    err=dc1394_video_set_transmission(camera,DC1394_OFF);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not stop the camera?\n");

    /*-----------------------------------------------------------------------
     *  close camera
     *-----------------------------------------------------------------------*/
    dc1394_video_set_transmission(camera, DC1394_OFF);
    dc1394_capture_stop(camera);
    dc1394_camera_free(camera);
    dc1394_free (d);
    return 0;
}
