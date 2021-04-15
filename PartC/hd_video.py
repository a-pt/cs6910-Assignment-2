from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import time
import cv2,os

classes=[]
fileptr = open('coco.names', 'r')
Lines = fileptr.readlines()
for line in Lines:
  classes.append(line[:-1])
print(len(classes))
print(classes)

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

layer = net.getLayerNames()
outputlayers = [layer[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors= np.random.uniform(0,255,size=(len(classes),3))

vs=FileVideoStream('traffic.mp4').start()#0 for 1st webcam
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0
frame_array = []
import os
from os.path import isfile, join

while True:
    frame = vs.read()
    frame_id+=1
    frame = imutils.resize(frame, width=500)
    (height,width) = frame.shape[:2]
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False) #reduce 416 to 320    

        
    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs[1])
    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for predict in out:
            scores = predict[5:]
            class_id = np.argmax(scores)
            if class_id==0:
                confidence = scores[class_id]
                if confidence > 0.3:
                    #onject detected
                    cx= int(predict[0]*width)
                    cy= int(predict[1]*height)
                    w = int(predict[2]*width)
                    h = int(predict[3]*height)

                    x=int(cx - w/2)
                    y=int(cy - h/2)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
            

    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
    
    cv2.imshow("Image",frame)
    frame_array.append(frame)
    key = cv2.waitKey(1) & 0xFF #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break;
pathOut='vedio.mp4'
size=(width,height)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    out.write(frame_array[i])
    out.release()
cap.release()    
cv2.destroyAllWindows()
