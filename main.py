# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:21:47 2022

@author: Grégoire
"""


import cv2
import numpy as np
from scipy.stats import mode
from tensorflow.keras.models import load_model

#load model from CNN to recognize a letter
model=load_model('model.h5')

#onehot
alphabet=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
       'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#shape of letter to work with CNN
width_cnn=35
height_cnn=35

#Transform the photo into a matrix
img=cv2.imread("awa.jpeg")
#Transform RGB into black/white
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#resize to have the same shape
img= cv2.resize(img, (1200,1000), interpolation = cv2.INTER_AREA)
width,height=img.shape

#Create a mask with a threshold
seuil=150
mask=np.zeros((width,height),np.uint8)
for i in range(width):
    for j in range(height):
        if img[i][j] <seuil:
            mask[i][j]=img[i][j]

#Dilate the mask to lose information and to have a global shape
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
dilate = cv2.dilate(mask, kernel, iterations=2)

#copy img black and white to extract the letter later 
image=img.copy()
image2=img.copy()

#Find contours of the words
cnts,h= cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image=img, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

#Save the coordinate of the bounding rectangle 
list_rect=[]
for c in cnts:
    x,y,w,h=cv2.boundingRect(c)

    #Select the rectangle tall enought
    if h >7:
        list_rect.append([x,y,w,h])
# sort the liist according y to have the line
list_rect=sorted(list_rect,key= lambda x:x[1])

# create a dict to save the coord of rectangle from each line
dico_ligne={}
ligne=1
for r in range(len(list_rect)):
        if ligne not in dico_ligne.keys():
            dico_ligne[ligne]=[]

        x=list_rect[r][0]
        y=list_rect[r][1]
        w=list_rect[r][2]
        h=list_rect[r][3]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        if r!=0:
            #if y+1>y+10 it's a new line
            if y-list_rect[r-1][1]>10:
                ligne+=1
                dico_ligne[ligne]=[]
        dico_ligne[ligne].append(list_rect[r])
print(dico_ligne.keys())

for k in dico_ligne.keys():
    with open("results.txt","a") as f:
        f.write("\n")
    m=0
    #Sort the rectangle according x
    dico_ligne[k]=sorted(dico_ligne[k])
    for r in dico_ligne[k]:    
        m+=1
        x=r[0]
        y=r[1]
        w=r[2]
        h=r[3]
        #Extract the word
        word=image2[y:y+h,x:x+w]
        cv2.imwrite("mot\ligne{}_mot_{}.jpg".format(k,m),word)
        with open("results.txt","a") as f:
            f.write(" ")
        width_word= word.shape[0]
        heigth_word=word.shape[1]
        #Add a pad to have a better mask
        pad=30
        new=np.zeros((width_word+pad,heigth_word+pad),dtype=np.uint8)
        new.fill(255)
        for i in range(width_word):
            for j in range(heigth_word):
                new[i+int(pad/2)][j+int(pad/2)]=word[i][j]

        image_word=new.copy()
        image2_word=new.copy()
        
        #Add blur to have a global shape
        new=cv2.blur(image_word, (2,2))
        # canny give edges
        canny_word=cv2.Canny(new,125,175)

        cnts_word,h= cv2.findContours(canny_word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=new, contours=cnts_word, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        n=1
        #save rectangle of letters
        list_lettres=[]
        for c in cnts_word:
            x,y,w,h = cv2.boundingRect(c)
    #ar = w / float(h)
            if h>6:
                cv2.rectangle(image_word, (x,y),(x+w,y+h),(0,0,0), 1)
                list_lettres.append([x,y,w,h])
                list_lettres=sorted(list_lettres)
        for r in range(len(list_lettres)):
                    x=list_lettres[r][0]
                    y=list_lettres[r][1]
                    w=list_lettres[r][2]
                    h=list_lettres[r][3]

                    letter=image2_word[y:y+h,x:x+w]
                    if letter.shape[0]<=width_cnn and letter.shape[1]<=height_cnn:
                        cv2.imwrite("lettre\ligne{}_mot{}_lettre{}.jpg".format(k,m,n),letter)
                        n+=1
                        padx=height_cnn-letter.shape[0]
                        pady=width_cnn-letter.shape[1]

                        pred=np.zeros((height_cnn,width_cnn))
                        pred.fill(255)
                        for i in range(padx//2):
                            pred[i,:]=255
                        for i in range(height_cnn-padx//2,height_cnn):
                            pred[i,:]=255
                        for j in range(pady//2):
                            pred[:,j]=255
                        for j in range(width_cnn-pady//2,width_cnn):
                            pred[:,j]=255
                        for i in range(letter.shape[0]):
                            for j in range(letter.shape[1]):
                                pred[i+padx//2,j+pady//2]=letter[i,j]
                        pred=pred/255
                        with open("results.txt","a") as f:
                            f.write(alphabet[np.argmax(model.predict(pred.reshape((1,height_cnn,width_cnn,1))))])