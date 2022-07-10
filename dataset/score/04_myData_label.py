
# _*_ coding:utf-8 _*_
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil
import cv2
 
# sets=[('myData', 'train'),('myData', 'val'), ('myData', 'test')]  # 根据自己数据去定义
sets=[('myData', 'train'),('myData', 'val')]  # 根据自己数据去定义

class2id = {
    'person':0, 'cat':1, 'dog':2, 'horse':3

}
 
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(year, image_id):
    in_file = open('./Annotations/%s.xml'%(image_id),encoding='utf-8')
    out_file = open('./labels/%s/%s.txt'%(year,image_id), 'w')
    # print(in_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    img = cv2.imread("./JPEGImages/"+image_id+".jpg")
    # shutil.copy("./JPEGImages/"+image_id+".jpg","./images/%s/%s.jpg"%(year,image_id))

    sp = img.shape

    h = sp[0] #height(rows) of image
    w = sp[1] #width(colums) of image
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls_ = obj.find('name').text
        if cls_ not in list(class2id.keys()):
            print("没有该label: {}".format(cls_))
            continue
        cls_id = class2id[cls_]
        xmlbox = obj.find('bndbox')
        x1 = float(xmlbox.find('xmin').text)
        x2 =  float(xmlbox.find('xmax').text)
        y1 = float(xmlbox.find('ymin').text)
        y2 = float(xmlbox.find('ymax').text)

        # 
        if x1 <= 0:
            x1 = 10
        if y1 <= 0:
            y1 = 10

        if x2 >= w:
            x2 = w-10
        if y2 >= h:
            y2 = h-10

        if x1 > x2:
            x1,x2 = x2,x1
        if y1 > y2:
            y1,y2 = y2,y1
        b = (x1, x2, y1 ,y2)
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
wd = getcwd()
 
for year, image_set in sets:
    if not os.path.exists('./labels/'):
        os.makedirs('./labels/')
    image_ids = open('./ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    # list_file = open('./%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        # list_file.write('%s/JPEGImages/%s.jpg\n'%(wd, image_id))  # 写了train或val的list
        convert_annotation(image_set, image_id)
    # list_file.close()


# labels/标注数据有了
# train val的list数据也有了
