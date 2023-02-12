import cv2 as cv
import numpy as np
import serial
import struct
import opencv_video


last_width = 0
last_length = 0
port = '/dev/ttyUSB0'  # 串口号
baudrate = 115200  # 波特率
ser = serial.Serial(port, baudrate, timeout=1)

yolo_tiny_model = "yolov4-tiny.weights";
yolo_tiny_cfg = "yolov4-tiny.cfg";
capture = cv.VideoCapture(0)

# Load names of classes
classes = None
with open("coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load tensorflow model
net = cv.dnn.readNetFromDarknet(yolo_tiny_cfg, yolo_tiny_model)
# set back-end
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

if (ser.isOpen() == True):
    print("串口打开成功")

def usart(data):
    ser.write(str(data).encode("UTF-8"))

#image = cv.imread('dog.jpg')
while (True) :
    classIds = []
    confidences = []
    boxes = []
    max_area = 0
    max_index = 0
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
    classIds,confidences,boxes = opencv_video.dnn(outs,w,h,classIds,confidences,boxes)

    # 使用非最大抑制
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    max_index = opencv_video.draw(indices,classes,boxes,image,classIds,max_area,max_index)

    try:
        if classes[classIds[max_index]] == "person":
            x , y, width, height = opencv_video.max_class(boxes,max_index)
            area = int((width + height) / 2)
            k = 575663
            y_length = k/area
            x_wide = (100/250)*(x-320)*(y_length/200)
            out_length = y_length*0.2+0.8*last_length
            out_width = x_wide*0.2+0.8*last_width
            last_length = out_length
            last_width = out_width
            data = struct.pack("<bbhh", 0x2C, 0x12, int(out_width/10), int(out_length/10))
            usart(data)
            print(out_width,out_length)
    except:
        print("no person")


    cv.namedWindow("YOLOv4-tiny-Detection-Demo", cv.WINDOW_NORMAL)
    cv.imshow('YOLOv4-tiny-Detection-Demo', image)
    c = cv.waitKey(30) & 0xff
    if c == 27:
        capture.release()
        break
cv.destroyAllWindows()
