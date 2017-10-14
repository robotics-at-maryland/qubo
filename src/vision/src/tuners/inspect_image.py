import cv2
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",
    help="path to the (optional) video file")
ap.add_argument("-b","--buffer",type=int,default=64,help="max buffer size")

args = vars(ap.parse_args() )
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video",False):
    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# from http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# press ctrl + c to close the window
while True:
    grabbed,frame=camera.read()
    if args.get("video") and not grabbed:
        break
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) and False:
        break

cap.release()
cv2.destrotAllWindows()
