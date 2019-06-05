# Instance Segmentation using Mask R-CNN
Blur background in live streaming using MASK R-CNN

## Project structure

1. __mask-rcnn-coco/__: Mask R-CNN model directory
    - __frozen_inference_graph.pb__: MASK R-CNN model weights (pre-traind on COCO dataset)
    - __mask_rcnn_inception_v2_coco_2018_01_28.pbtxt__: MASK R-CNN model configuration
    - __object_detection_classes_coco.txt__: all 90 classes are listed in this file, one per line

2. __instance_segmentation.py__: script to run our program
## How to run

```
python instance_segmentation.py --mask-rcnn mask-rcnn-coco --kernel 41 
```