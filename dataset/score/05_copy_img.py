import os
import shutil


with open("./ImageSets/Main/val.txt",'r') as f:
    lines = f.readlines()
for line in lines:
    print(line.strip())

    img_file = line.strip() + ".jpg"
    print(img_file)

    shutil.copy("./JPEGImages/"+img_file,"./images/val/"+img_file)


with open("./ImageSets/Main/train.txt",'r') as f:
    lines = f.readlines()
for line in lines:
    print(line.strip())

    img_file = line.strip() + ".jpg"
    print(img_file)

    shutil.copy("./JPEGImages/"+img_file,"./images/train/"+img_file)