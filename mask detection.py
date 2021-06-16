import numpy as np
import cv2
#%matplotlib notebook
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('nose.xml')
cap = cv2.VideoCapture(0)

def show(image_show):
    plt.axis('off')
    plt.imshow(image_show)  #灰色:plt.imshow(image_show,'gray)
    plt.show()
    
while(cap.isOpened()):
    ret, img = cap.read()
    if ret==True:
        
        color = (0,255,0) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #轉換印出照片 gray: BGR2GRAY
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3)
        nose = nose_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(40,40))
        
        if (len(nose)!=0):
             color = (0,0,255)
        
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            show(roi_gray)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        
        # Display the resulting frame
        cv2.imshow('frame',img)
        #if cv2.waitKey(10) & 0xFF == ord('q'):
        if cv2.waitKey(10) == 27:
            cap.release()
            break
    else:
        print("No video")
        cap.release()
        cv2.destroyAllWindows()
        quit()
    # When everything done, release the capture

cv2.destroyAllWindows()
cap.release()
cap.release()