#!/usr/bin/env python

from __future__ import print_function, division
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import thread
import numpy as np
from Queue import LifoQueue
# receiving the msg with type CompressedImage
# http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
# receiving the msg with type Image
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

class camera_receiver(object):

    def __init__( self,name, topic, msgtype, verbose=False ):
        self.topic = topic
        self.msgtype = msgtype
        self.name = name

        self.bridge = CvBridge()
        self.convertor = self.convertor_query()
        self.verbose = verbose
        self.subthread = None
        self.ros_image = LifoQueue( maxsize = 5 )

    def callback(self,image):
        if self.ros_image.full():
            #self.ros_image.get()
            self.ros_image.empty()
        self.ros_image.put( image )

    def __call__(self):
        def monitor(threadnam):
            # disable_signals could set to be true in the future
            #rospy.init_node(self.name,anonymous=True)
            rospy.Subscriber(self.topic, self.msgtype, self.callback )
        #monitor()
            #rospy.spin()
        self.subthread = thread.start_new_thread( monitor,(self.name,) )

    def convertor_query(self):
        if self.msgtype == Image:
            def bridge(image,**args):
                return self.bridge.imgmsg_to_cv2(image,**args)
            return bridge
        elif self.msgtype == CompressedImage:
            def bridge(image,**args):
                img_array_1d = np.fromstring(image.data,np.uint8)
                cv_img = cv2.imdecode(img_array_1d, 1)#cv2.CV_LOAD_IMAGE_COLOR)
                return cv_img
            return bridge
        else:
            print("the msgtype for this receiver in only support sensor_msgs.msg.Image and sensor_msgs.msg.CompressedImage")
            self.__del__()

    def spit(self,**args):
        # desired_encoding="passthrough"
        if self.verbose : print( 'received image of type: "%s"' % ros_data.format)
        screen = self.convertor(self.ros_image.get(),**args)
        return screen

    def __del__(self):
        print( "the camera_receiver {} was deleted".format(self.name) )


if __name__ == '__main__':
    print("For testing")
    rospy.init_node('testingcam_rec',anonymous=True)
    cam_r_i= camera_receiver('IMAGE_ONE', '/qubo_test/camera/camera_image', Image)
    cam_r_ci= camera_receiver('CIMAGE_ONE','/qubo_test/camera/camera_image/compressed', CompressedImage)

    cam_r_ci() # start Subscribe to topic
    cam_r_i()  # start Subscribe to topic

    print('press e to exit')
    # https://stackoverflow.com/questions/14494101/using-other-keys-for-the-waitkey-function-of-opencv
    # https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=waitkey
    while True:
        cv2.imshow( 'Image', cam_r_i.spit( desired_encoding="bgr8") )
        cv2.imshow( 'CompressedImage', cam_r_ci.spit( ) )
        if cv2.waitKey(1) == ord('e'): break

    cv2.destroyAllWindows()
