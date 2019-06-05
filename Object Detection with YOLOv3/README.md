# Object Detection using YOLOv3

## Project structure

* __yolo-coco/__: the YOLOv3 object detector pre-trained (on the COCO dataset) model file, trained by [Darknet team](https://pjreddie.com/darknet/yolo/).
* __images/__: folder containing input images for testing and evaluation purposes
* __videos/__: folder containing input videos for testing and evaluation purposes
* __output/__: folder containing output images/videos processed by YOLO 

## How to run

* Run YOLO object detection in videos

```
python yolo_realtime.py --input videos/car_chase_01.mp4 --output output/car_chase_01.avi --yolo yolo-coco
```

* Run YOLO object detection in images

```
python yolo.py --image images/soccer.jpg --yolo yolo-coco
```

## Demo 
![](outputImage.png)
