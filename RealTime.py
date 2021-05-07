import cv2
import numpy as np
import face_recognition
import os

path = 'Training Images'
images = []
classnames = []
mylist = os.listdir(path)
for cls in mylist:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classnames.append(os.path.splitext(cls)[0])


# print(classnames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)
    return encodeList


encodeListKnow = findEncodings(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDist=face_recognition.face_distance(encodeListKnow,encodeFace)
        print(faceDist)
        matchIndex=np.argmin(faceDist)

        if matches[matchIndex]:
            name=classnames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

# faceLoc=face_recognition.face_locations(imgRohit)[0]
# encodeRohit=face_recognition.face_encodings(imgRohit)[0]
#
# faceLocTest=face_recognition.face_locations(imgTest)[0]
# encodeRohitTest=face_recognition.face_encodings(imgTest)[0]
# #print(faceLoc)
# cv2.rectangle(imgRohit,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# results=face_recognition.compare_faces([encodeRohit],encodeRohitTest)
# faceDist=face_recognition.face_distance([encodeRohit],encodeRohitTest)
