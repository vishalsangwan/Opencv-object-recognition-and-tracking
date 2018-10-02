import cv2
import numpy as np
import os 
import sys
import time

#tracker = cv2.TrackerCSRT_create()
#tracker = cv2.TrackerMIL_create()
#tracker = cv2.TrackerBoosting_create()
#tracker = cv2.TrackerKCF_create()
#tracker = cv2.TrackerTLD_create()
#tracker = cv2.TrackerMedianFlow_create()
tracker = cv2.TrackerMOSSE_create()








ok = False

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\\vishal sangwan\\Desktop\\pwm camera\\meetup\\trainer\\trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'vishal',"mummy"] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
#minW = 0.1*cam.get(3)
#minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically
    if ok == False:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            #minSize = (int(minW), int(minH)),
            )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
                
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                bbox = (x,y,w,h)
                break
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
        
            #cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    ok = tracker.init(img,bbox)

    #start timer
    timer = cv2.getTickCount()
    #update tracker
    ok, bbox = tracker.update(img)
    # Calculate Frames per second
    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
        cv2.rectangle(img,p1,p2,(255,0,0),2)
    else:
        cv2.putText(img, "Tracking failure detected and detecting face again", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        ok = False
        time.sleep(1)
        
            


    cv2.putText(img, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    cv2.imshow("image",img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()