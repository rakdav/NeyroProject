import cv2
img=cv2.imread('face.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_cascade=(cv2.
              CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml'))
faces=face_cascade.detectMultiScale(gray,1.3,5)
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('img',img)
cv2.waitKey()