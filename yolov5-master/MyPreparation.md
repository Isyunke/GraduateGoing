
# 下载yolov5
-- git clone https://github.com/ultralytics/yolov5.git
或者在 https://github.com/ultralytics/yolov5 上下载zip包
-- 安装requirement.txt文件

# 测试自带的权重文件
```shell
python .\detect.py --source .\data\images --weights .\weight\yolov5n-7-k5.pt
```

# 收集数据集
从自然环境中收集数据集，但是图片最好具有多样性，采集在不同的天气、不同的时间、不同的光照强度、不同角度、不同来源的图片。
具体要求可搜索：YOLO官方推荐数据集需求。

# 标记数据集
使用labelImg 标记数据集，生成label

# 训练数据集
```shell
!python train.py --batch-size 4 --epochs 100 --data ../datasets/archive/person.yaml --weights weights/yolov5n.pt --cache ram
```

# 测试
```shell
python .\detect.py --source C:\Users\gf66\Pictures\luoye --weights runs/train/exp15/weights/best.pt
```

# 训练的比较好的一个weight
```shell
python .\detect.py --source C:\Users\gf66\Pictures\luoye --weights runs/train/leaf_det_model/weights/best.pt
```

# 如果模型没有运行完被中断了，使用一下代码进行结果验证图的生成
```shell
!python val.py --data data/mask_data.yaml --weights runs/train/exp_yolov5s/weights/best.pt --img 640
```

# 最终检测，使用detect.py进行检测，包含一下命令
```shell
 # 检测摄像头
 python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source 0  # webcam
 # 检测图片文件
  python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source file.jpg  # image 
 # 检测视频文件
   python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source file.mp4  # videos
 # 检测一个目录下的文件
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt path/  # directory
 # 检测网络视频
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'https://youtu.be/NUsoVlDFqZg'  # YouTube videos
 # 检测流媒体
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream                            

```