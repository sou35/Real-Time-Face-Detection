import cv2
import random
# Load some pre-trained data on face frontals from opencv(hear cascode algorithm)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#To capture video from webcam
webcam=cv2.VideoCapture(0)
#Iterate forever over frames
while True:
    #Read the Current frame
    successful_frame_read,frame =webcam.read()
    #Must convert to grayscale
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detect face
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
    
    # Draw rectangles around the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),  0),2)
    
    
    
    cv2.imshow('Clever Programmer Face Detector',frame)
    cv2.waitKey(1)

