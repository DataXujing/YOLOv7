import argparse
import time
from pathlib import Path
import os
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


half = False
save_img = True
save_txt = True

conf_thres = 0.25
iou_thres=0.45

weights = "./runs/train/yolov7/weights/last.pt"

test_path = "./test_img"

device = select_device("0")
model = attempt_load(weights, map_location=device)  # load FP32 model

if half:
    model.half()  # to FP16

# stride = int(model.stride.max())  # model stride
# imgsz = check_img_size(imgsz, s=stride)  # check img_size

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


files = os.listdir(test_path)

for file in files:
    img_path = os.path.join(test_path,file)
    img0 = cv2.imread(img_path) 
    img = letterbox(img0,new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    
    pred = model(img, augment=False)[0]

    # Apply NMS
    # (center x, center y, width, height) 
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False)


    # Process detections
    for i, det in enumerate(pred):  # detections per image
        save_path = os.path.join("./res",file)
        txt_path = os.path.join("./res/labels",file[:-4])
  
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round() # 映射到原图，这个可以在NMS后做

            # # Print results
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class
            #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    line = (cls, *xywh, conf)  # label format

                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        #print(f'{s}Done. ({t2 - t1:.3f}s)')

        # # Stream results
        # if view_img:
        #     cv2.imshow(str(p), im0)
        #     cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, img0)
    print(file)
            




