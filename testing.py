import cv2
import numpy as np
from collections import defaultdict
import os

""" It loads the pre-trained weights and configuration file for YOLO using the cv2.dnn.readNet() function.
It reads the video from a file named '***' using the cv2.VideoCapture() function.
It defines the class of object to detect as 'person'
It enters into a while loop that reads the frames of the video one by one and performs object detection on each frame.
It converts the frame to a blob and passes it as an input to the model
It gets the output layers name and forward the blob through the network.
It detects the bounding box of the detected object, the class of the object and the confidence score.
It applies non-maxima suppression to eliminate multiple overlapping bounding boxes.
It draws the bounding boxes on the frame and write the class of the object and confidence score on the imag"""

net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
path = "data_target_tracking"
video_path_list = sorted([f for f in os.listdir(path) if f.endswith('.mp4')])
x_ = 0
y_ = 0
st_track_line = []
en_track_line = []
xs, ys = 0, 0
xe, ye = 0, 0
for i, v_path in enumerate(video_path_list):
    cap = cv2.VideoCapture(path + '/' + v_path)
    classes = ['person']
    frameSize = (640, 512)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'output07\output_video{i}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, frameSize)
    centroid_dict = defaultdict(list)
    object_id_list = []

    while 1:
        ret, img = cap.read()
        # img = cv2.resize(img, (640, 640))
        if ret is False:
            break

        hight, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)

        output_layers_name = net.getUnconnectedOutLayersNames()

        layerOutputs = net.forward(output_layers_name)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3] * hight)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3] * hight)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
        font = cv2.FONT_HERSHEY_PLAIN
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                x_ = x + 1
                y_ = y + 1
                confidence = str(round(confidences[i], 2))
                cv2.rectangle(img, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)
                cv2.putText(img, 'Person' + " " + confidence, (x, y + 1), font, 1.5, (0, 0, 255), 2)
                centroid_dict[i].append((x, y))
                if i not in object_id_list:
                    object_id_list.append(i)
                    start_pt = (x, y)
                    end_pt = (x, y)
                    cv2.line(img, start_pt, end_pt, (255, 0, 0), 2)
                else:
                    l = len(centroid_dict[i])
                    for pt in range(len(centroid_dict[i])):
                        if not pt + 1 == l:
                            start_pt = (centroid_dict[i][pt][0], centroid_dict[i][pt][1])
                            end_pt = (centroid_dict[i][pt + 1][0], centroid_dict[i][pt + 1][1])
                            st_track_line.append(start_pt)
                            en_track_line.append(end_pt)
                            cv2.line(img, start_pt, end_pt, (255, 0, 0), 2)
        else:
            if len(st_track_line) > 0 and len(en_track_line) > 0:
                # for start_pt, end_pt in zip(st_track_line, en_track_line):
                cv2.rectangle(img, (x_ - 10, y_ - 10), (x_ + 10, y_ + 10), (0, 255, 0), 2)
                cv2.putText(img, 'Person' + " " + '0.0', (x_, y_ + 1), font, 1.5, (0, 0, 255), 2)
                for start_pt, end_pt in zip(st_track_line, en_track_line):
                    cv2.line(img, start_pt, end_pt, (255, 0, 0), 2)

        out.write(img)
        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()
cv2.destroyAllWindows()
print('done')
