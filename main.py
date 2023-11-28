import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import numpy as np


model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        # print(colorsBGR)

        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('wrongway.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
tracker=Tracker()
area1=[(593,227),(602,279),(785,274),(774,220)]
area2=[(747,92),(785,208),(823,202),(773,95)]
wup={}
wrongway=[]

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))


    results=model.predict(frame)
    # print(results)
    a=results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")
    """
        px contains the bounding box coordinates of all the objects detected in the frame
    """
    # print("px ", px)
    list=[]
            
    for index,row in px.iterrows():
        # print("Index = ", index, " row = ", row)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d] # Car is 2.000 in coco.txt
        if 'car' in c: 
            list.append([x1,y1,x2,y2])

    bbox_idx=tracker.update(list)
    
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=x3
        cy=y4
        # Marking the bottom left corner of the bounding box
        is_car_in_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        # print(is_car_in_area1) 

        # Store the ID's of the cars moving from area 1 
        if(is_car_in_area1 >= 0):
            wup[id] = (cx, cy)

        # Checking for the same car if it ever goes to area 2
        if id in wup:
            is_car_in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if is_car_in_area2 >= 0:

                cv2.circle(frame,(cx,cy),8,(255,0,0),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)

                if id in wrongway:
                    continue
                else:
                    wrongway.append(id)
                    print(f"Car {id} is going in the wrong way")
        
    print(wup)

    total_wrongway_cars = len(wrongway)

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,255,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,255,255),2) 
    cv2.putText(frame, f"Wrongway cars: {total_wrongway_cars}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
