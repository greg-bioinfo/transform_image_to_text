# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 11:02:51 2022

@author: Grégoire
"""

import cv2
import numpy as np
from scipy.stats import mode
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

img=cv2.imread("text.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img= cv2.resize(img, (1200,1000), interpolation = cv2.INTER_AREA)
width,height=img.shape

print(img.shape)
seuil=150

mask=np.zeros((width,height),np.uint8)
for i in range(width):
    for j in range(height):
        if img[i][j] <seuil:
            mask[i][j]=img[i][j]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
dilate = cv2.dilate(mask, kernel, iterations=1)


image=img.copy()
image2=img.copy()

cnts,h= cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image=img, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

list_rect=[]
for c in cnts:
    x,y,w,h=cv2.boundingRect(c)

    #ar = w / float(h)
    if h >7:
        list_rect.append([x,y,w,h])
list_rect=sorted(list_rect,key= lambda x:x[1])
print(list_rect)

ligne=1
dico_ligne={}
for r in range(len(list_rect)):
        if ligne not in dico_ligne.keys():
            dico_ligne[ligne]=[]

        x=list_rect[r][0]
        y=list_rect[r][1]
        w=list_rect[r][2]
        h=list_rect[r][3]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        if r!=0:
            if y-list_rect[r-1][1]>10:
                ligne+=1
                dico_ligne[ligne]=[]
        dico_ligne[ligne].append(list_rect[r])
print(dico_ligne.keys())

for k in dico_ligne.keys():
    i=0
    dico_ligne[k]=sorted(dico_ligne[k])
    for r in dico_ligne[k]:    
        i+=1
        x=r[0]
        y=r[1]
        w=r[2]
        h=r[3]
        word=image2[y:y+h,x:x+w]
        cv2.imwrite("mot\ligne{}_mot_{}.jpg".format(k,i),word)
        #data = pytesseract.image_to_string(word, lang='fra',config='--psm 6')
        #print(data)
"""
    
    data = pytesseract.image_to_string(word, lang='fra',config='--psm 6')
    with open("texte.txt","a",encoding="utf-8") as f:
        f.write(str(data))

cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    
    
    ar = w / float(h)
    if ar < 5:
        cv2.drawContours(dilate, [c], -1, (255,0,0), 2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

result = 255 - cv2.bitwise_and(dilate, mask)
data = pytesseract.image_to_string(result, lang='eng',config='--psm 6')
print(data)
"""
#cv2.imshow("words",result)
#cv2.imshow("res",dilate)
cv2.imshow("img",img)
cv2.imshow("dilate",mask)
cv2.imshow("image",image)
k=cv2.waitKey() & 0XFF
if k==ord("q"):
    cv2.destroyAllWindows()

