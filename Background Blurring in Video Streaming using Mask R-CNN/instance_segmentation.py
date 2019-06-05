import imutils
from imutils.video import VideoStream
import numpy as np 
import time
import cv2
import os
import argparse

# construct the argument parse and parse the arguments
# --mask_rcnn: path to the Mask R-CNN directory (i.e., mask-rcnn-coco)
# --confidence: minimum probabilit to filter out week detections
# --threshold: minimum threshold for pixel-wise mask segmentation
# --kernel: kernel size of Gaussian blur kernel
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask_rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-k", "--kernel", type=int, default=41,
	help="size of gaussian blur kernel")
args = vars(ap.parse_args())

# load the COCO class labels 
labelsPath = os.path.sep.join([args["mask_rcnn"], 
                            "object_detection_classes_coco.txt"])
labels = open(labelsPath).read().split('\n')

# paths for Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
                            "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load pre-trained Mask R-CNN on COCO dataset
print("Loading Mask R-CNN...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# construct kernel for Gaussian blur and set privacy mode
kernel = (args["kernel"], args["kernel"])
privacy = False 

# initialize the video stream
print("Starting video stream...")
vs = VideoStream(src=0).start()

# set a small amount of time to allow sensor to warm up
time.sleep(2.0)

# loop over frames from the video file stream
while True:
    # retrieve the frame from the threaded video stream
    frame = vs.read()

    # resize the frame
    frame = imutils.resize(frame, width=600)
    (H, W) = frame.shape[:2]

    # construct a blob from the input image 
    # then perform forward pass of the Mask R-CNN
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    # sort the indexes of the bounding boxes
    # by the corresponding prediction probability (in descending order)
    idxs = np.argsort(boxes[0, 0, :, 2])[::-1]

    # initial the mask, ROI and coordinated of the person for current frame
    mask = None
    roi = None
    coords = None

    # loop over the indexes 
    for idx in idxs:
        # extract the class ID and confidence associated with the confidence
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # if the detection is not the 'person' class, ignore it
        if labels[classID] != 'person':
            continue

        # filter the weak predictions by ensuring the detected probability
        # is greated than the minumum probability
        if confidence > args["confidence"]:
            # scale the bounding box corrdinates
            # back to the relative dim of the image
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])

            # copmpute width and height of the bounding box
            (startX, startY, endX, endY) = box.astype("int")
            coords = (startX, startY, endX, endY)
            boxW = endX - startX
            boxH = endY - startY 

            # extract the pixel-wise segmentation for the object
            mask = mask[idx, classID]
            # resize the mask to have the same dim of the bounding box
            mask = cv2.resize(mask, (boxW, boxH),
                            interpolation=cv2.INTER_NEAREST)
            # threshold to create a binary mask
            mask = (mask > args["threshold"])

            # extract the ROI
            roi = frame[startY:endY, startX:endX][mask]
            break

    # initialize the output frame
    output = frame.copy()

    # check if the mask is not None and in privace mode
    if mask is not None and privacy:
        # blur the output frame
        output = cv2.Gaussianblur(output, kernel, 0)

        # add the ROI to the output frame for only the masked region
        (startX, startY, endX, endY) = coords
        output[startY:endY, startX:endX][mask] = roi

        # show the output frame
        cv2.imshow("Video Call", output)
        key = cv2.waitKey(1) & 0xFF
    
    # if the `p` key was pressed, toggle privacy mode
    if key == ord("p"):
        privacy = not privacy

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
vc.stop()





