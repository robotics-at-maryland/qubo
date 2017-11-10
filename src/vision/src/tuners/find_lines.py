import cv2
import argparse, sys
import numpy as np

# grab the video from the args
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help = "path to the video file")

args = vars(ap.parse_args())

if args.get("video", False):
    video = cv2.VideoCapture(args["video"])
else:
    print("Missing video")
    sys.exit(0)

def process_img(frame):
    red = frame.copy()
    red[:,:,0] = 0
    # red[:,:,0] = 0
    # return frame
    gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    # thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    _, thr = cv2.threshold(red[:,:,2], 105, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # check out opencv find contours
    im2, contours, heirarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    eps = 0.005 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
    return frame

while True:
    if not video.isOpened():
        video.open()
        continue
    r, frame = video.read()
    if frame is not None:
        final = process_img(frame)
        cv2.imshow('ret', final)
        # cv2.imshow('frame',frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
