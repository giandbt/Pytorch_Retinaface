import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from PIL import Image

def get_faces(txt_path, padding=0.4, save_dir = './data/widerface/faces'):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get image paths and face labels
    imgs_path = []
    words = []
    f = open(txt_path, 'r')
    lines = f.readlines()
    isFirst = True
    labels = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if isFirst is True:
                isFirst = False
            else:
                labels_copy = labels
                words.append(labels_copy)
                labels = []
            path = line[2:]
            path = txt_path.replace('label.txt', 'images/') + path
            imgs_path.append(path)
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)
    
    words.append(labels)
    
    count = 0
    for index in range(len(imgs_path)):
        print('Processing image %i out of %i' %(index, len(imgs_path)))
        img = Image.open(imgs_path[index]).convert("RGB")
        name = os.path.basename(imgs_path[index])
        labels = words[index]
        if len(labels) == 0:
            pass
        for idx, label in enumerate(labels):
            if sum(label[0:4]) == 0:
                count += 1
                continue
                
            # bbox
            x1 = label[0]
            y1 = label[1]
            x2 = label[0] + label[2]
            y2 = label[1] + label[3]
            h = label[3]
            w = label[2]
            pad = padding / 2 # for each side
            
            if label[2] <= 0 or label[3] <= 0:
                count += 1
                continue
            
            x1 = max(0, x1-w*pad)
            y1 = max(0, y1-h*pad)
            x2 = min(x2+w*pad, img.size[0])
            y2 = min(y2+h*pad, img.size[1])
            
            face_img = img.crop((x1, y1, x2, y2))
            face_img.save(os.path.join(save_dir, name[:-4] + '_%i.jpg'%idx))
        
    print('All faces saved successfully. Faces skipped: %i'%count)
if __name__ == '__main__':
    txt_path = './data/widerface/train/label.txt'
    get_faces(txt_path)
    