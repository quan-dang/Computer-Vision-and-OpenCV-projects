# Image and Video Color Recover

## Prerequisites
- Load the colorization Caffe models and prototxt, then put into models folder (mkdir models)
    https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt 

    http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel 

    http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2_norebal.caffemodel 

- Load the cluster centers
    https://github.com/richzhang/colorization/blob/master/colorization/resources/pts_in_hull.npy?raw=true

## How to run the program

* Run colorizeImage.py only for images  

```
python colorizeImage.py --input <img path>
```

* Run colorizeVideo.py only for videos  

```
python colorizeVideo.py --input <video path>
```


