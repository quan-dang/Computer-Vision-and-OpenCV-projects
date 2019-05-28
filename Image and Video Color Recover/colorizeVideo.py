import numpy as np
import cv2
import argparse
import os.path

parser = argparse.ArgumentParser(description="Colorize GrayScale Video")
parser.add_argument('--input', help="Path to the video.")
args = parser.parse_args()

if args.input == None:
    print("Please give the grayscale image path!")
    print("Usage: python colorizeImage.py --input GrayscaleVideo.mp4")
    exit()


if os.path.isfile(args.input) == 0:
    print("Input file does not exist!")
    exit()

# read the input video
cap = cv2.VideoCapture(args.input)
hasFrame, frame = cap.read()

# params for output video
output = args.input[:-4] + "_colorized.avi"
video_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (frame.shape[1], frame.shape[0]))

# specify the paths for the 2 models files
protoFile = "./models/colorization_deploy_v2.prototxt"
weightsFile = "./models/colorization_release_v2.caffemodel"
# weightsFile = "./models/colorization_release_v2_norebal.caffemodel"

# load the cluster centers
pts_in_hull = np.load("./pts_in_hull.npy")

# read the network
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# populate cluster centers as 1x1 convolution kernel
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)

net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# input shape
W_in = 224
H_in = 224

while cv2.waitKey(1):
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)

    if not hasFrame:
        break


    img_rgb = (frame[:, :, [2,1,0]] * 1.0 / 255).astype(np.float32)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_l = img_lab[:, :, 0]

    # resize lightness channnel to network input size
    img_l_rs = cv2.resize(img_l, (W_in, H_in))
    img_l_rs = img_l_rs - 50 # mean centering

    net.setInput(cv2.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0, :, :].transpose((1,2,0)) # get ab channels result

    # get original image size
    (H_original, W_original) = img_rgb.shape[:2]
    ab_dec_us = cv2.resize(ab_dec, (W_original, H_original))
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis], ab_dec_us), axis=2) # concatenate with the original image L
    img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

    video_writer.write((img_bgr_out * 255).astype(np.uint8))


video_writer.release()


print('Colorized video has been saved as ' + output)
print("\nDONE!")