import cv2
import numpy as np
from object_detection import ObjectDetection
import math


#Initialize Object Detection 
od = ObjectDetection()


cap = cv2.VideoCapture('los_angeles.mp4')

#Intialize Count
count=0 
center_points_prev_frame=[]
tracking_objects={}
track_id=0

while True:
    ret, frame = cap.read()
    count+=1
    if not ret:
        break
    
    #center point of current frame
    center_points_curr_frame = []
    
    #Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x,y,w,h) = box
        cx = int((x + x + w)/2)
        cy = int((y+y+h)/2)
        center_points_curr_frame.append((cx,cy))
        print("Frame",count,x,y,w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  
    if count<=2:
        for pt in center_points_curr_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])
                if distance < 20:
                    tracking_objects[track_id]=pt
                    track_id+=1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_curr_frame_copy = center_points_curr_frame.copy()
        
        for obj_id,pt2 in tracking_objects_copy.items():
            obj_exists=False
            for pt in center_points_curr_frame_copy:
                distance = math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])
                
                #update obj_poistion
                if distance <20: 
                    tracking_objects[obj_id]=pt
                    obj_exists=True
                    if pt in center_points_curr_frame:
                        center_points_curr_frame.remove(pt)
                    continue
            
            #remove the id 
            if not obj_exists: 
                tracking_objects.pop(obj_id)
    
        for pt in center_points_curr_frame:
            tracking_objects[track_id]= pt
            track_id+=1                  

    for obj_id,pt in tracking_objects.items():
        cv2.circle(frame,pt,5,(0,0,255),-1)
        cv2.putText(frame, str(obj_id), (pt[0],pt[1]-7), 0, 1, (0,0,255), 1)
        
    print("CURR FRAME",center_points_curr_frame)
    print("PREV FRAME",center_points_prev_frame)
    
    cv2.imshow("Frame",frame)
    
    #making copy of center points 
    center_points_prev_frame = center_points_curr_frame.copy()  
    
    key = cv2.waitKey()
    if key==27:
        break 

cap.release()
cv2.destroyAllWindows() 