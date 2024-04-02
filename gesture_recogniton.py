import numpy as np
import math
import cv2
import time
from cvzone.HandTrackingModule import HandDetector

height, width = 480,640

#initialize the frame and set parameters
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4, height)
detector = HandDetector(maxHands = 1)

offset = 20
imgsize = 300

#set the folder for saving images
folder = "/Users/aser8929/Data/Z"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        #set the size of the frame
        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgwhite = np.ones((imgsize,imgsize,3), np.uint8)*255
        imgcropshape = imgcrop.shape

        aspectratio = h/w
        
        #if height is bigger than width - put white bars on the sides
        if aspectratio>1:
            k=imgsize/height
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop,(wcal,imgsize))
            imgresizeshape = imgresize.shape
            wgap = math.ceil((imgsize-wcal)/2)
            imgwhite[:, wgap:wcal+wgap] = imgresize

        #if width is bigger than height - put white bars on top and bottom
        else:
            k=imgsize/width
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop,(imgsize,hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize-hcal)/2)
            imgwhite[hgap:hcal+hgap,:] = imgresize

        #display the images on the screen
        cv2.imshow("imgcrop", imgcrop)
        cv2.imshow("imgwhite", imgwhite)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    
    #when "s" is pressed, save the current frame to the folder
    if key == ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/image_{time.time()}.png',imgwhite)
        print(counter)