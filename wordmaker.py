import numpy as np
import math
from cvzone.ClassificationModule import Classifier
import cv2
import time
from cvzone.HandTrackingModule import HandDetector
import keras

height, width = 480,640

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4, height)
detector = HandDetector(maxHands = 1)
#import model and labels file
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

offset = 20
imgsize = 300

counter = 0
global index, previndex
sentence = ""
index = None
previndex = None
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgwhite = np.ones((imgsize,imgsize,3), np.uint8)*255
        imgcropshape = imgcrop.shape

        aspectratio = h/w

        if aspectratio>1:
            k=imgsize/height
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop,(wcal,imgsize))
            imgresizeshape = imgresize.shape
            wgap = math.ceil((imgsize-wcal)/2)
            imgwhite[:, wgap:wcal+wgap] = imgresize
            #get pridiction from the model using the current frame
            previndex = index
            prediction, index = classifier.getPrediction(img)
            print(prediction,index)

        else:
            k=imgsize/width
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop,(imgsize,hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize-hcal)/2)
            imgwhite[hgap:hcal+hgap,:] = imgresize

        cv2.imshow("imgcrop", imgcrop)
        cv2.imshow("imgwhite", imgwhite)
        

    cv2.imshow("image", img)
    cv2.waitKey(1)
##########making the words using the prediction#######
    if previndex != index:
        sentence = sentence + labels[index]