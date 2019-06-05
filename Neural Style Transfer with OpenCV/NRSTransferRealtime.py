# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2
 
# construct the argument parser and parse the arguments
# --models: directory containing NRS transfer models
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
	help="path to directory containing neural style transfer models")
args = vars(ap.parse_args())

# retrieve the paths to all NRS transfer models (i.e, models)
modelPaths = paths.list_files(args["models"], validExts=(".t7"))
modelPaths = sorted(list(modelPaths))

# generate unique IDs for each of the model paths
# and combine the 2 lists together
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

# loop over all the model paths
modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)

# load the NRS transfer model
print("Loading model...")
net = cv2.dnn.readNetFromTorch(modelPath)

# initialize the video stream 
print("Starting video stream...")
vs = VideoStream(src=0).start()

# set a small amount of time for the camera sensor to warm up
time.sleep(2.0)

print("{}. {}".format(modelID + 1, modelPath))

# loop over frames from the video file stream
while True:
    # retrieve the frame from the video stream
    frame = vs.read()

    # resize the frame
    frame = imutils.resize(frame, width=600)

    # create a copy of the frame
    orig = frame.copy()

    # grab the image dims
    (h, w) = frame.shape[:2]

    # construct a blob from the frame, set the input 
    # and perform a forward pass of the network
    blob = cv2.dnn.blobFromImage(frame, 1.9, (w, h),
                            (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    # reshape the output tensor, add back in the mean subtraction
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0

    # swap the channel ordering
    output = output.transpose(1, 2, 0)

    # show the original frame along with the output NRS transfer
    cv2.imshow("Input", frame)
    cv2.imshow("Output", output)

    key = cv2.waitKey(1) & 0xFF

	# if the `n` key is pressed (for "next"), load the next neural
	# style transfer model
    if key == ord("n"):
        # grab the next neural style transfer model model and load it
        (modelID, modelPath) = next(modelIter)
        print("[INFO] {}. {}".format(modelID + 1, modelPath))
        net = cv2.dnn.readNetFromTorch(modelPath)
 
	# otheriwse, if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

