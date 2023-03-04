import cv2 as cv
import numpy as np

def find_smallest(x_length):
    if len(x_length) == 0:
        return None
    min_value = x_length[0]
    for num in x_length:
        if abs(num) < abs(min_value):
            min_value = num
    return min_value

def find_sides(x_length,center_column):
    if len(x_length) == 0:
        return None,None
    left_side = []
    right_side = []    
    for num in x_length:
        if num < center_column:
            left_side.append(num)
        if num > center_column:
            right_side.append(num)
    return left_side,right_side



yolo_tiny_model = "myyolov4_best.weights";
yolo_tiny_cfg = "myyolov4.cfg";
capture = cv.VideoCapture(2)


# Load names of classes
classes = None
with open("myvoc.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load tensorflow model
net = cv.dnn.readNetFromDarknet(yolo_tiny_cfg, yolo_tiny_model)
# set back-end
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#image = cv.imread('dog.jpg')
while (True) :
    ref, image = capture.read()

    h, w = image.shape[:2]

    blobImage = cv.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), None, True, False);
    outNames = net.getUnconnectedOutLayersNames()
    net.setInput(blobImage)
    outs = net.forward(outNames)

    t, _ = net.getPerfProfile()
    fps = 1000 / (t * 1000.0 / cv.getTickFrequency())
    label = 'FPS: %.2f' % fps
    cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    # 绘制检测矩形
    classIds = []
    confidences = []
    boxes = []
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

    # 使用非最大抑制
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    x_length = []
    left_side = []
    right_side = []
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        x = int((box[0] + box[2])/2)
        cv.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 2, 8, 0)
        cv.putText(image, classes[classIds[i]], (left, top),cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        x_length.append(x-160)
    try:
        center_column = find_smallest(x_length)
        left_side,right_side = find_sides(x_length,center_column)
        left_column = find_smallest(left_side)
        right_column = find_smallest(right_side)
        print(x_length,center_column,left_column,right_column)
        mode = input("请输入您想识别的柱子：")
        if mode == 1:
            print(center_column)
        if mode == 2:
            print(left_column)
        if mode == 3:
            print(right_column)
    except:
        print("no column")

    cv.namedWindow("YOLOv4-tiny-Detection-Demo", cv.WINDOW_NORMAL)
    cv.imshow('YOLOv4-tiny-Detection-Demo', image)
    c = cv.waitKey(30) & 0xff
    if c == 27:
        capture.release()
        break
cv.destroyAllWindows()
