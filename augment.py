# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import torch
import PIL
from PIL import Image

path=str(os.getcwd())
count=1
path=path+'\\Ctrain'
x=list(os.listdir(path))
#print(x)
count=1
for nam in x:
    #print(nam)
    p=str(path)+'\\'+nam
    #print(p)
    y=p+"\\images"
    z=p+"\\masks"
    #print(y)
    if(not os.path.isdir(y)):
        continue
    if(not os.path.isdir(z)):
        continue   
    p1=os.listdir(y)
    p2=os.listdir(z)
    
    for c in p1:
        pth1=y+'\\'+c
        pth2=z+'\\'+c
        #print(y)
        pth2=pth2[0:-3]+'png'
        #print(pth1)
        #print(pth2)
        im1= Image.open(pth1)
        im2= Image.open(pth2)
        
        out1 = im1.transpose(Image.FLIP_LEFT_RIGHT)
        out2 = im2.transpose(Image.FLIP_LEFT_RIGHT)
        out1.save(y+'\\'+str(count)+'.jpg')
        out2.save(z+'\\'+str(count)+'.png')
        count=count+1
        
        out3 = im1.transpose(Image.ROTATE_90)
        out4 = im2.transpose(Image.ROTATE_90)
        out3.save(y+'\\'+str(count)+'.jpg')
        out4.save(z+'\\'+str(count)+'.png')
        count=count+1           
 
        out5 = im1.transpose(Image.ROTATE_270)
        out6 = im2.transpose(Image.ROTATE_270)
        out5.save(y+'\\'+str(count)+'.jpg')
        out6.save(z+'\\'+str(count)+'.png')
        count=count+1 
        
        out7 = im1.transpose(Image.ROTATE_180)
        out8 = im2.transpose(Image.ROTATE_180)
        out7.save(y+'\\'+str(count)+'.jpg')
        out8.save(z+'\\'+str(count)+'.png')
        count=count+1         
