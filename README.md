##  Official YOLOv7 训练自己的数据集（端到端TensorRT 模型部署）

目前适用的版本： `v0.1` 

> 我只能说太卷了!

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

<img src="./figure/performance.png" height="480">


#### Performance 

MS COCO

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |

### 1.训练环境配置

```shell

# create the docker container, you can change the share memory size if you have more. # 注意这个share memory尽量设置的大一点，训练的话！
sudo nvidia-docker run --name yolov7 -it -v /home/myuser/xujing/hackathon2022:/yolov7 --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# go to code folder
cd /yolov7
```


### 2.训练集准备

YOLOv7类似于YOLOR支持YOLOv5类型的标注数据构建，如果你熟悉YOLOv5的训练集的构造过程，该部分可以直接跳过，这里我们提供了构建数据集的代码，将数据集存放在`./dataset`下：

```shell
./datasets/score   # 存放的文件，score是数据集的名称
├─images           # 训练图像，每个文件夹下存放了具体的训练图像
│  ├─train
│  └─val
└─labels           # label，每个文件夹下存放了具体的txt标注文件，格式满足YOLOv5
    ├─train
    └─val
```

我们提供了VOC标注数据格式转换为YOLOv5标注的具体代码，存放在`./dataset`下，关于YOLOv5的标注细节可以参考：
+ https://github.com/DataXujing/YOLO-v5
+ https://github.com/DataXujing/YOLOv6#2%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87


### 3.模型结构或配置文件修改

+ 1.修改模型的配置文件

i.训练数据的配置:`./data/score.yaml`

```shell
train: ./dataset/score/images/train # train images
val: ./dataset/score/images/val # val images
#test: ./dataset/score/images/test # test images (optional)

# Classes
nc: 4  # number of classes
names: ['person','cat','dog','horse']  # class names
```

ii.模型结构的配置:`./cfg/training/yolov7_score.yaml`

```shell
# parameters
nc: 4  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [12,16, 19,36, 40,28]  # P3/8
  - [36,75, 76,55, 72,146]  # P4/16
  - [142,110, 192,243, 459,401]  # P5/32

...

```


### 4.模型训练

```shell
python3 train.py --workers 8 --device 0 --batch-size 32 --data data/score.yaml --img 640 640 --cfg cfg/training/yolov7_score.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# 一机多卡
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/score.yaml --img 640 640 --cfg cfg/training/yolov7_score.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

```


### 5.模型评估

```shell
python3 test.py --data data/score.yaml --img 640 --batch 32 --conf 0.001 --iou 0.45 --device 0 --weights ./runs/train/yolov7/weights/last.pt --name yolov7_640_val

```

### 6.模型推断

```shell
python3 inference.py

```

推断效果如下：

<div align=center>

|                                 |                                  |                                  |                                  
| :-----------------------------: | :------------------------------: | :------------------------------: | 
| <img src="figure/image1.jpg"  height=270 width=270> | <img src="figure/image2.jpg" width=270 height=270> | <img src="figure/image3.jpg" width=270 height=270> |  

</div>



### 7.TensorRT 模型加速

我们实现了端到端的TensorRT模型加速，将NMS通过Plugin放入TensorRT Engine中。

1. YOLOv7转ONNX

注意：如果直接转ONNX会出现`scatterND`节点，可以做如下修改

<div align=center>
<img src="figure/onnx_0.png" >
</div>

```shell
python3 ./models/export.py --weights ./runs/train/yolov7/weights/last.pt --img-size 640 640 --grid

```

<div align=center>
<img src="figure/onnx.png" >
</div>

1. YOLOv7的前处理

YOLOv6, YOLOv7的前处理和YOLOv5是相同的。

其C++的实现如下：

```c++
void preprocess(cv::Mat& img, float data[]) {
	int w, h, x, y;
	float r_w = INPUT_W / (img.cols*1.0);
	float r_h = INPUT_H / (img.rows*1.0);
	if (r_h > r_w) {
		w = INPUT_W;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_H - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = INPUT_H;
		x = (INPUT_W - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	//cudaResize(img, re);
	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

	int i = 0;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = out.data + row * out.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}

}

```


3. YOLOv7的后处理

不难发现，YOLOv5, YOLOv6, YOLOv7的后处理基本是完全相同的，TensorRT后处理的代码基本可以复用

增加后处理和NMS节点：

```shell

python tensorrt/yolov7_add_postprocess.py
python tensorrt/yolov7_add_nms.py

```

onnx中增加了如下节点：

<div align=center>
<img src="figure/onnx_1.png" >
</div>


4. TRT序列化Engine

```shell
trtexec --onnx=last_1_nms.onnx --saveEngine=yolov7.engine --workspace=3000 --verbose
```

5. `VS2017`下编译运行`./tensorrt/yolov7`



<div align=center>

|                                 |                                  |                                  |                                  
| :-----------------------------: | :------------------------------: | :------------------------------: | 
| <img src="figure/image1_trt.jpg"  height=270 width=270> | <img src="figure/image2_trt.jpg" width=270 height=270> | <img src="figure/image3_trt.jpg" width=270 height=270> |  

</div>


