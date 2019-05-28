import cv2 as cv
import os
import argparse
import pytesseract

try:
    import Image
except ImportError:
    from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# construct the argument parse and parse the arguments
# --image: path to the input image
# --preprocess (optional): preprocessing method, thresh (default) or blur
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
            help="path to the input image")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
            help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the image and grayscale the image
img = cv.imread(args["image"])
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# check if we need to preprocess the image by "thresh"
if args["preprocess"] == "thresh":
    grayimg = cv.threshold(grayimg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

# check if we need to process the image by median "blur"
if args["preprocess"] == "blur":
    grayimg = cv.medianBlur(grayimg, 3)


# write the grayscale image to disk as a temp file to apply OCR on it
filename = "{}.png".format(os.getpid())
cv.imwrite(filename, grayimg)

# load load rthe image, apply OCR and delete the temp file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
cv.imshow("Image", img)
cv.imshow("Output", grayimg)
cv.waitKey(0)