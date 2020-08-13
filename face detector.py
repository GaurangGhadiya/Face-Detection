import cv2

face_cascade = cv2.CascadeClassifier("C:\\Users\\Dell\\AppData\\Local\\Programs\\Python\\Python38-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

img = cv2.imread("C:\\Users\\Dell\\Pictures\\Camera Roll\\image.jpg",1)

grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

shapes = face_cascade.detectMultiScale(grey_img, scaleFactor=1.05, minNeighbors=5)

for x,y,w,h in shapes:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

resized = cv2.resize(img,(int(img.shape[1]/2), int(img.shape[0]/2)))

cv2.imshow('Face Detection', resized)

cv2.waitKey(0)

cv2.destroyAllWindows()