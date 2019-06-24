import cv2
import threading
import numpy as np
import math
from tkinter import *
import webbrowser
import time
import nltk
import speech_recognition as sr
import os
import video_test
start=Tk()
fd=cv2.CascadeClassifier("/root/adhoc/darknet/frontal.xml")
enable_music=False
enable_camera=False
enable_voice=False
enable_dect=False

def enableMusic():
    global enable_music
    enable_music=True
def opencam():
    global enable_music
    global enable_camera
    global enable_voice
    global enable_dect
    cap = cv2.VideoCapture(0)
    num=0
     
    while(1):
        ret, frame = cap.read()
        cam_frame=frame
        if enable_camera==True:
            face=fd.detectMultiScale(frame,1.15,5)
            for x,y,w,h in face:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            if len(face)>=3:
                cv2.imwrite('capturqe.jpg',cam_frame) 
                enable_camera=False

            else:
                pass

        elif enable_voice==True:
            r=sr.Recognizer()
            with sr.Microphone() as source:
                print("Give Voice Command: ")
                audio=r.listen(source)
            try:
                sentence=str(r.recognize_google(audio))
                print("Google Speech Recognition thinks you said: "+sentence )
                token=nltk.word_tokenize(sentence)
                token=[i.lower() for i in token]
                if 'open' in token:
                    if token[token.index('open')+1] == 'facebook':
                        webbrowser.get('/usr/bin/firefox').open_new_tab('https://www.facebook.com')
                    elif token[token.index('open')+1] == 'google':
                        webbrowser.get('/usr/bin/firefox').open_new_tab('https://www.google.com')
                    else:
                        print("Unknown command.")
                    enable_voice=False
                else:
                    print("Unknown Command")
            except sr.UnknownValueError:
                print("Google Speech Recognition could not Understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service: {0}".format(e))

        elif enable_dect==True:
            video_test.start_dect(cap)
            enable_dect=False
        else:
            try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
            
                
                frame=cv2.flip(frame,1)
                kernel = np.ones((3,3),np.uint8)

                #define region of interest
                roi=frame[100:350, 100:350]

                cv2.rectangle(frame,(100,100),(350,350),(0,255,0),0)    
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # define range of skin color in HSV
                lower_skin = np.array([0,20,70], dtype=np.uint8)
                upper_skin = np.array([20,255,255], dtype=np.uint8)

                 #extract skin colur imagw  
                mask = cv2.inRange(hsv, lower_skin, upper_skin)

                #extrapolate the hand to fill dark spots within
                mask = cv2.dilate(mask,kernel,iterations = 4)

                #blur the image
                mask = cv2.GaussianBlur(mask,(5,5),100) 

                #find contours
                contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

               #find contour of max area(hand)
                cnt = max(contours, key = lambda x: cv2.contourArea(x))

                #approx the contour a little
                epsilon = 0.0005*cv2.arcLength(cnt,True)
                approx= cv2.approxPolyDP(cnt,epsilon,True)


                #make convex hull around hand
                hull = cv2.convexHull(cnt)

                 #define area of hull and area of hand
                areahull = cv2.contourArea(hull)
                areacnt = cv2.contourArea(cnt)

                #find the percentage of area not covered by hand in convex hull
                arearatio=((areahull-areacnt)/areacnt)*100

                 #find the defects in convex hull with respect to hand
                hull = cv2.convexHull(approx, returnPoints=False)
                defects = cv2.convexityDefects(approx, hull)

                # l = no. of defects
                l=0

                #code for finding no. of defects due to fingers
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    pt= (100,180)


                    # find length of all sides of triangle
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    s = (a+b+c)/2
                    ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

                    #distance between point and convex hull
                    d=(2*ar)/a

                    # apply cosine rule here
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57


                    # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                    if angle <= 90 and d>30:
                        l += 1
                        cv2.circle(roi, far, 3, [255,0,0], -1)

                    #draw lines around hand
                    cv2.line(roi,start, end, [0,255,0], 2)


                l+=1

                #print corresponding gestures which are in their ranges
                font = cv2.FONT_HERSHEY_SIMPLEX
                if l==1:
                    if areacnt<2000:
                        cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    else:
                        if arearatio<27:
                            cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        else:
                            cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

                elif l==2:
                    cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                   # print(n)
                    #if n==0:
                    if enable_music==True:
                        webbrowser.get("/usr/bin/firefox").open_new_tab("https://www.youtube.com/watch?v=zdXiSlRrgWQ")
                        enable_music=False
                        timer=threading.Timer(5.0,enableMusic)
                        timer.start()
                    #time.sleep(2)
                      #  n=n+1
                       # num=num+1



                elif l==3:
                    cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    enable_camera=False 

                elif l==4:
                    cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    enable_voice=False

                elif l==5:
                    cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    enable_dect=True
                    #os.system("python3 video_test.py")
                else :
                    cv2.putText(frame,'Reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

                #show the windows
                cv2.imshow('mask',mask)            
            except:
                pass
        cv2.imshow('frame',frame)
        if cv2.waitKey(5) & 0xFF==ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()

butt=Button(start,text="Open Camera",command=opencam)
butt.pack()
start.geometry("300x300")
start.mainloop()
