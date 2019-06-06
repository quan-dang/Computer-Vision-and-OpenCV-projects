import cv2

# load cascade classifiers for face and eyepair detection
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eyepair_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_eyepair_big.xml")

# load glasses image
img_glasses = cv2.imread("images/glasses.png", -1) # -1 for alpha channel (if exists)

# create the mask for the classes
img_glasses_mask = img_glasses[:, :, 3]

#　convert glasses image to BGR, discard alpha channel
img_glasses = img_glasses[:, :, 0:3]

# load the input image to apply glasses filter
# face = cv2.imread("face.jpg")

# create VideoCapture object to get frames from webcam
vs = cv2.VideoCapture(0)

# loop over frames in the video file stream
while True:
    # read the next frame from the video
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, it means 
    # we have reached the end of the stream
    if not grabbed:
        print("No detected webcam!")
        break

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # loop over face detections
    for (x, y, w, h) in faces:
        # create the ROIs
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = frame[y:y + h, x:x + w]

        # detect the eyepair inside the detected face
        eyepairs = eyepair_cascade.detectMultiScale(roi_gray)

        # loop over the eyepairs detections (inside the detected face)
        for (ex, ey, ew, eh) in eyepairs:
            # calculate the coordinates where the glasses will be placed
            x1 = int(ex - ew / 10) 
            x2 = int((ex + ew) + ew / 10)
            y1 = int(ey)
            y2 = int(ey + eh + eh / 2)

            if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
                continue
            
            # calculate the width and height of the image with the glasses
            img_glasses_res_width = int(x2 - x1)
            img_glasses_res_height = int(y2 - y1)

            # resize the mask to be equal to the region where the glasses will be placed
            mask = cv2.resize(img_glasses_mask, (img_glasses_res_width, img_glasses_res_height))

            # create the inverse of the mask
            mask_inv = cv2.bitwise_not(mask)

            # resize the img_glasses to the previously calculated size
            img = cv2.resize(img_glasses, (img_glasses_res_width, img_glasses_res_height))

            # take the ROI from the BGR image
            roi = roi_color[y1:y2, x1:x2]

            # create ROI background and ROI faceground
            roi_background = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi_foreground = cv2.bitwise_and(img, img, mask=mask)

            # add roi_background and roi_foreground to create the result
            result = cv2.add(roi_background, roi_foreground)

            # set result into the color ROI
            roi_color[y1:y2, x1:x2] = result

            break


    # display the resulting frame
    cv2.imshow('Snapchat-based OpenCV glasses filter', frame)

    # Press any key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do some cleanup stuffs
cv2.destroyAllWindows()
vs.release()

        
 


    