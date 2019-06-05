import numpy as np 
import cv2 
import matplotlib.pyplot as pylab
import imutils
import argparse
import datetime

# construct the argument parse and parse the arguments
# --image: path to the input image
# --win_stride: step size in the x and y direction of our sliding windows
# --padding: amounts of pix the ROI is padded before HOG feature extraction and SVM classification
# --scale: control the scale of the image pyramid, allow us to detect people in images at multiple scales
# --mean_shift: apply mean shift grouping to the detectd bounding boxes
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--win_stride", type=str, default="(8, 8)",
	help="window stride")
ap.add_argument("-p", "--padding", type=str, default="(16, 16)",
	help="object padding")
ap.add_argument("-s", "--scale", type=float, default=1.05,
	help="image pyramid scale")
ap.add_argument("-m", "--mean_shift", type=int, default=-1,
	help="whether or not mean shift grouping should be used")
args = vars(ap.parse_args())

# evaluate the command line arguments
winStride = eval(args["win_stride"])
padding = eval(args["padding"])
meanShift = True if args["mean_shift"] > 0 else False

# create HOG descriptor using default people (pedestrian) detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# load the image and resize it
image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(400, image.shape[1]))

# start to record elapsed time
startTime = datetime.datetime.now()

# run detection using a spatial stride of 4 pixels (horizontal and vertical), a scale stride of 1.02
(boundingBoxes, weights) = hog.detectMultiScale(image, winStride=winStride, padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)

# elapsed time
print("Elapsed time: {}s".format((datetime.datetime.now() - startTime).total_seconds()))

# draw the original bounding boxes
for (x, y, w, h) in boundingBoxes:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Detections", image)
cv2.waitKey(0)









