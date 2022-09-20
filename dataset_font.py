from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import os
import string

alphabet=list(string.ascii_lowercase)+list(string.ascii_uppercase)#+ list(string.punctuation)
print(alphabet)
c=0
for i in alphabet:                
    for file in os.listdir(r"C:\Windows\Fonts"):
        if file.endswith("ttf"):
            c+=1
            #print(file)
            image=Image.new(mode="L", size=(35, 35),color =255)
            draw=ImageDraw.Draw(image)
            font=ImageFont.truetype("C:\Windows\Fonts\{}".format(file), 27)
            text="{}".format(i)
            draw.text((10, 0), text, font=font, fill=(0))
            image=np.array(image).reshape(35, 35, 1)
            #if os.path.isdir(r"C:\Users\Grégoire\Documents\programmation\opencv\font\{}".format(i)) == False:
                #os.mkdir(r"C:\Users\Grégoire\Documents\programmation\opencv\font\{}".format(i))
            cv2.imwrite("data_font\{}_{}.jpg".format(i,c),image)
            #im1 = draw.save(".\font\font_{}_{}.jpg".format(i,file))
            key=cv2.waitKey()
            if key&0xFF==ord('q'):
                quit()
                                                                    

