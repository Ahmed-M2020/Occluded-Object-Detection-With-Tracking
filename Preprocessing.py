"""This script is written in Python and uses the OpenCV library to open and process video frames. The purpose of the
script is to extract frames from a video, convert them to HSV color space and apply a binary mask to classify objects
in the frame. The script starts by importing the necessary libraries, cv2, numpy, os and pickle. The script then
declares several global variables that will be used later in the script: data_list, X, Y, cX, cY, frame_count. The
script then sets the video path and opens the video using cv2.VideoCapture(). The script uses a while loop to read
the video frame by frame. For each frame, the script converts the frame to the HSV color space using the
cv2.cvtColor() function. Then, the script creates a lower and upper bound for object classification using the numpy
array function. A binary mask is created using the cv2.inRange() function based on the lower and upper bounds. The
script applies a threshold to the mask using cv2.threshold() function and finds the contours of the objects using
cv2.findContours() function. The script then sorts the contours by area and calculates the center of each object
using the cv2.moments() function. The script then checks if the center of the object has changed and if so,
it saves the frame and the binary mask to the data_list. The script also increments the frame count. After the while
loop, the script closes all windows, saves the data_list to a pickle file and prints "done" to indicate that the
script has completed its execution. """

import cv2
import numpy as np
import os
import pickle

data_list = []
X, Y, cX, cY = 0, 0, 0, 0
frame_count = 0

data_dir = os.path.join('data_target_tracking', 'video_014.mp4')  # chamge Video name
cap = cv2.VideoCapture(data_dir)
path = "data_target_tracking"

while True:

    ret, frame = cap.read()
    if ret is False:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_b = np.array([0, 0, 191])
    u_b = np.array([240, 240, 240])
    mask = cv2.inRange(hsv, l_b, u_b)
    ret, thresh = cv2.threshold(mask, 155, 255, cv2.THRESH_BINARY)
    # !!! uncomment this line only with noisy videos

    # blur = cv2.blur(thresh, (5, 5))

    thresh = cv2.adaptiveThreshold(thresh, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 23, 5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours0 = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for objectId, cont in enumerate(contours0):
        M = cv2.moments(cont)
        if M['m00'] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

    if X != cX and Y != cY:
        print(cX, cY)
        frame_array = frame[:, :, 0]
        frame_array = np.expand_dims(frame_array, axis=2)
        mask_binary = np.expand_dims(thresh, axis=2)
        print(mask_binary.shape, frame_array.shape)
        data_list.append([frame_array, mask_binary])
        cv2.imshow("maskb", mask_binary)
        cv2.imshow("frameb", frame_array)
        frame_count += 1
        X, Y = cX, cY

    key = cv2.waitKey(6)
    if key == 27:
        break

cv2.destroyAllWindows()
print(data_dir, frame_count)
with open("014.pkl", "wb") as f:
    pickle.dump(data_list, f)

print("done")
