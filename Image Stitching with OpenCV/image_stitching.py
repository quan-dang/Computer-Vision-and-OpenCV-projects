from imutils import paths
import numpy as np 
import imutils
import cv2 as cv
import argparse

# construct the argument parser and parse the arguments
# --images: the directory contains images for stitching
# --output: the path contains the output image
# --crop: decide whether to crop out the largest rectangular region after stitching
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
            help="path to the directory of the input images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
            help="path to the output image")
ap.add_argument("-c", "--crop", type=int, default=0,
            help="whether to crop out the largest rectangular region")
args = vars(ap.parse_args())


# get the paths to the input images and initialize the images list
print("Loading images...")
imgPaths = sorted(list(paths.list_images(args["images"])))
imgs = []

# loop through image paths, load each one and add them to imgs to stitch
for imgPath in imgPaths:
    img = cv.imread(imgPath)
    imgs.append(img)


print("Stitching images...")
stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()
(status, stitched) = stitcher.stitch(imgs)

# check if status is '0', then OpenCV successfully performed stitching
if status == 0:
    # check to see whether crop out the largest rectangular region from the stitched image
    if args["crop"] > 0:
        # create a 10-pixel border surrounding the stitched image
        print("Croping the stitched image...")
        stitched = cv.copyMakeBorder(stitched, 10, 10, 10, 10,
                    cv.BORDER_CONSTANT, (0, 0, 0)) # (0, 0, 0) is the value to fill the border pixels

        # grayscale the stitched image and threshold it
        # such that all pixels > 0 are set to 255, and 0 for others
        grayimg = cv.cvtColor(stitched, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(grayimg, 0, 255, cv.THRESH_BINARY)[1]

        # retrieve all external contours in the threshold image, 
        # then find the largest contour, which will the the contour of the stitched image
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        # create the mask for the rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # create two copies of the mask:
        # one to serve as the actual minimum rectangular region
        # the other to serve as a counter for how many pixels to remove to form the minimum retangular region
        minRect = mask.copy()
        sub = mask.copy()

        # keep looping until there are no non-zero pixels left in the subtracted image
        while cv.countNonZero(sub) > 0:
            # erode the minimum rectangular mask and then subtract 
            # the thresholded image from the minimum rectangular mask
            # so we can count if there are any non-zero pixels

            # minRect will be reduced slowly until it can fit inside the inner part of the panorama 
            minRect = cv.erode(minRect, None)

            # it is used to decide whether to keep reducing the size of minRect
            # subtract thresh from minRect
            sub = cv.subtract(minRect, thresh)

        # find contours in the minimum rectangular mask
        # and extract the bounding box (x, y) - coordinates
        cnts = cv.findContours(minRect.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        (x, y, w, h) = cv.boundingRect(c)

        # use the bounding box coordinates to extract the final stitched image
        stitched = stitched[y:y+h, x:x+w]

    # write the output stitched image to disk
    cv.imwrite(args["output"], stitched)

    # display the output stitched image to the screen
    cv.imshow("Stitched", stitched)
    cv.waitKey(0)

else:
    print("Image stitching failed {}".format(status))
