import cv2 as cv
import numpy as np
import serial
import struct

def max_class(boxes,max_index):
    box = boxes[max_index]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    x = int((2 * left + width) / 2)
    y = int((2 * top + height) / 2)
    return x,y,width,height

def dnn(outs,w,h,classIds,confidences,boxes):
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # numbers are [center_x, center_y, width, height]
            if confidence > 0.05:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    return classIds,confidences,boxes

def draw(indices,classes,boxes,image,classIds,max_area,max_index):
    for i in indices:
        if classes[classIds[i]] == "person":
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 2, 8, 0)
            cv.putText(image, classes[classIds[i]], (left, top),cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            area = width * height
            if area > max_area:
                max_area = area
                max_index = i
    return max_index



