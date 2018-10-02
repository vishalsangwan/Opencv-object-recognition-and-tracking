#importing the libraries
import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cap.set(3,640)# set width
cap.set(4,480)# set Height

while True:
    ret, img = cap.read()
    #img = cv2.flip(img,-1) #fliiping the video
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors =5)

    #iterating the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break


    cv2.imshow("video",img)

    
    
    

cap.release()
cv2.destroyAllWindows()

