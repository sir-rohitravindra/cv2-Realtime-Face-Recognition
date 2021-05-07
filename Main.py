import cv2
#import numpy as np
import face_recognition


imgRohit=face_recognition.load_image_file("Training Images/Rohit1.jpg")
imgRohit=cv2.cvtColor(imgRohit,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file("Training Images/Rohit1.jpg")
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgRohit)[0]
encodeRohit=face_recognition.face_encodings(imgRohit)[0]

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeRohitTest=face_recognition.face_encodings(imgTest)[0]
#print(faceLoc)
cv2.rectangle(imgRohit,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeRohit],encodeRohitTest)
faceDist=face_recognition.face_distance([encodeRohit],encodeRohitTest)
print(results,faceDist)

cv2.putText(imgTest,f'{results},{round(faceDist[0],2)}',(50,50),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)

cv2.imshow('Rohit Ravindra',imgRohit)
cv2.imshow('Rohit Test',imgTest)
cv2.waitKey(0)
