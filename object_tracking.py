import cv2
import time
import math

video = cv2.VideoCapture("bb3.mp4")
#Load Tracker
tracker = cv2.TrackerCSRT_create()
#Read the first frame
returned, img = video.read()
#Select the bounding box on the image
bbox = cv2.selectROI("Tracking", img, False)
#Initialize the tracker
tracker.init(img, bbox)
print(bbox)

def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]) 
    cv2.rectangle(img, (x, y), ((x+w),(y+h)), (255,0,255), 3, 1)
    cv2.putText(img, "Tracking", (75,90), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 2, 3)


while True:
    check, img = video.read()  

    success, bbox = tracker.update(img)
    if(success):
        drawBox(img,bbox)
    else:
        cv2.putText(img, "Lost", (75,90), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 2, 3, True)

    cv2.imshow("result",img)
            
    key = cv2.waitKey(50)

    if key == 32:
        print("Stopped!")
        break


video.release()
cv2.destroyALLwindows()