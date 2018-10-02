# importing the libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

#loading images and videos from local storage
#img = cv2.imread("C:\\Users\\vishal sangwan\\Desktop\\pwm camera\\meetup\\lena_color_512.tif")
#cv2.imshow("img",img)
#print(img.shape)



#matplotlib type showing image
#img_matplot = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(img_matplot)
#plt.show()


#video loading from local storage

video = cv2.VideoCapture("C:\\Users\\vishal sangwan\\Desktop\\pwm camera\\meetup\\demo.mp4")

while 1:
    ret, frame = video.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()

cv2.waitKey(0) #wait for any key press
cv2.destroyAllWindows()





