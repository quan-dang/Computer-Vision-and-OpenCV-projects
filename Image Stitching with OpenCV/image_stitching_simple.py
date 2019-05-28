from imutils import paths
import numpy as np 
import imutils
import cv2 as cv
import argparse

# construct the argument parser and parse the arguments
# --images: the directory contains images for stitching
# --output: the path contains the output image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
            help="path to the directory of the input images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
            help="path to the output image")
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
    # write the output stitched image to disk
    cv.imwrite(args["output"], stitched)

    # display the output stitched image to the screen
    cv.imshow("Stitched", stitched)
    cv.waitKey(0)

else:
    print("Image stitching failed {}".format(status))
