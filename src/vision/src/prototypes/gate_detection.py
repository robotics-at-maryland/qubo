#! /usr/bin/env python2
import cv2
import argparse
import numpy as np
import math

parser = argparse.ArgumentParser(description='Gate detection prototype')
parser.add_argument('image', type=str, help='Image file to examine')
args = parser.parse_args()

image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, None, 50, 10)

# image = gray
# for x1, y1, x2, y2 in lines[0]:
#     print x1
#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def is_vertical(x):
    x1, y1, x2, y2 = x[0]
    angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
    if (angle < 115 and angle > 65) or (angle > -115 and angle < -65):
        return True
    return False


verts = [x for x in lines if is_vertical(x)]
means = [(l[0][0] + l[0][2]) / 2 for l in verts]

mean = lambda x: sum(x) / len(x)
avg = mean(means)
left = mean([x for x in means if x < avg])
right = mean([x for x in means if x >= avg])
print means
if lines is not None:
    for i in range(0, len(lines)):
        x1, y1, x2, y2 = lines[i][0]

        angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if (angle < 115 and angle > 65) or (angle > -115 and angle < -65):
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.line(image, (left, 0), (left, 1000), (0, 0, 255), 3)
    cv2.line(image, (right, 0), (right, 1000), (0, 0, 255), 3)
    cv2.line(image, ((left + right)/2, 0), ((left + right)/2, 1000), (0, 255, 0), 3)

print(image.shape)
cv2.imshow('lines', image)
cv2.waitKey(0)
