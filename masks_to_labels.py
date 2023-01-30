import pickle
import cv2
"""reads the data from a pickle file named "combined_data.pkl" which contains a list of images and corresponding 
masks. It iterates through the data and resizes each image and mask to a 640x640 resolution using the cv2.resize() 
function. For each image, it finds the contours of the mask using the cv2.findContours() function. It sorts the 
contours by their area using the lambda function and cv2.contourArea(). For each contour, it finds the bounding 
rectangle using the cv2.boundingRect() function, which returns the (x, y) coordinates of the top-left corner and the 
width and height of the rectangle. It computes the center point (cX, cY) of the bounding rectangle by adding half the 
width and half the height to the top-left corner coordinates. It opens a file named "yolo_data/count.txt" and writes 
the class label, center coordinates and width and height of the bounding rectangle as a single line. It saves the 
image as "yolo_data/count.jpg". """

data = []
X, Y = 0, 0
with open('combined_data.pkl', 'rb') as f:
    file = pickle.load(f)
    for pkl in file:
        for f in pkl:
            data.append(f)

# for im in data:
count = 0
for i in range(0, len(data)):
    frame = cv2.resize(data[i][0], (640, 640), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(data[i][1], (640, 640), interpolation=cv2.INTER_LINEAR)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours0 = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for objectId, cont in enumerate(contours0):
        x, y, w, h = cv2.boundingRect(cont)

        # Find center points
        cX, cY = x + w / 2, y + h / 2
        if X != cX and Y != cY:
            file = open(f"yolo_data/{count}.txt", "w")
            file.write(f'{0} {cX/640} {cY/640} {w/640} {h/640}')
            cv2.imwrite(f"yolo_data/{count}.jpg", frame)
            count += 1
            X, Y = cX, cY
    cv2.waitKey(0)  # wait for ay key to exit window
cv2.destroyAllWindows()
