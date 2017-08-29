import cv2
import numpy as np 
import imutils
cap = cv2.VideoCapture('gestures.mp4')

lower = np.array([0, 133, 77], dtype = "uint8")
upper = np.array([255, 173, 127], dtype = "uint8")

lower2 = np.array([54, 133, 77], dtype = "uint8")
upper2 = np.array([163, 173, 127], dtype = "uint8")

face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
while(True):

    ret, frame = cap.read()
    
    frame = imutils.resize(frame, width = 800)
    #face removal
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0), thickness = cv2.FILLED)

    #bg subtract
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skinMask = cv2.inRange(converted, lower, upper)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    #canny
    edges = cv2.Canny(skin,40,80)
    _,contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours (skin, contours, (0, 0, 255), (0, 0, 255), 2, 2, cv2.FILLED)
    cv2.drawContours(skin, contours, -1, (0,255,0), 2)

    cv2.imshow("images", skin)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
