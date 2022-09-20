# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:04:49 2022

@author: Grégoire
"""

import cv2
import numpy as np
import pytesseract
import os

img=cv2.imread(".\mot\ligne2_mot_11.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

width=img.shape[0]
heigth=img.shape[1]
pad=30

new=np.zeros((width+pad,heigth+pad),dtype=np.uint8)
new.fill(255)
for i in range(width):
    for j in range(heigth):
        new[i+int(pad/2)][j+int(pad/2)]=img[i][j]

image=new.copy()
image2=new.copy()

new=cv2.blur(image, (2,2))
    
canny=cv2.Canny(new,125,175)


cnts,h= cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image=new, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
i=0
list_lettres=[]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    #ar = w / float(h)
    if h>6:
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,0), 1)
        list_lettres.append([x,y,w,h])
list_lettres=sorted(list_lettres)
for r in range(len(list_lettres)):
    x=list_lettres[r][0]
    y=list_lettres[r][1]
    w=list_lettres[r][2]
    h=list_lettres[r][3]
    
    word=image2[y:y+h,x:x+w]
    cv2.imwrite("lettre\lettre{}.jpg".format(i),word)
    i+=1
cv2.imshow("image",new)
cv2.imshow("dilate",image)
k=cv2.waitKey() & 0XFF
if k==ord("q"):
    cv2.destroyAllWindows()



"""
for files in os.listdir(".\mot"):
    img=cv2.imread(".\mot\{}".format(files))
    ret,img=cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    data = pytesseract.image_to_string(img, lang='fra',config='--psm 6')
    with open("liste_mot.txt","a",encoding="utf-8") as f:
        f.write(data)
"""