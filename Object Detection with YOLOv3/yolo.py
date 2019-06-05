import numpy as np 
import cv2 
import os
import time 
import argparse

# construct the argument parse and parse the arguments
# --image: the path to the input image
# --yolo: the base path to the YOLO directory (i.e., yolo-coco)
# --confidence: minimum probability to filter weak detections
# --threshold: non-maxima suppression threshold  
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
labels = open(labelsPath).read().strip().split('\n')

# initialize a list of colors to represent each possible class label
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# retrieve the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO model trained on COCO dataset (80 classes)
print("Loading YOLO...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load the input image
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]

# determine the output layer names we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image, set the input 
# then perform a forward pass of the YOLO object detector
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# start to record elapsed time
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show elapsed time
print("Elapsed time: {}s".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and classIDs
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the classID and confidence of the current detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by confidence
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative 
            # to the size of the image
            box = detection[0:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype('int')

            # use the centerX and centerY to derive the top and left
            # corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences and classIDs 
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maximum suppression to reduce number of bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

# ready to draw box and class text on the image
# ensure there exists at least one detection 
if len(idxs) > 0:
    # loop over the idxs we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# show the output
cv2.imshow("Image", image)
cv2.waitKey(0)
