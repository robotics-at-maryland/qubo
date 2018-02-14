import numpy as np
import torch

def evaluate( boxes , gt_boxes ):
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        #timer.event('get region boxes')


        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)

            total = total + num_gts

            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1
        #timer.event('evaluate')
        #timer.calculate()
        #timer.report(mode='rec',release=True)
    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
