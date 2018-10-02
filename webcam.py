#importing libraries
import numpy as np
import cv2 


def main():
    #initialising the webcam
    video = cv2.VideoCapture(0) 

    while video.isOpened():
        ret, frame = video.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()