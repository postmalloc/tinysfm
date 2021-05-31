# tinysfm
<img src="./demo.gif" width="350px"/>  

TinySFM is a tiny (~50 loc) implementation of Incremental Structure From Motion. There are no optimisations, no bundle adjustment, no real frame picking strategies. It is small, simple, and (hopefully) easy to grok!

Expects an OpenMVG dataset-like directory structure -
```
dataset/
    Herz-Jesus-P8/
        img0.jpg
        img1.jpg
        img2.jpg
        ..
        K.txt
```
`K.txt` must contain the camera parameters in a 3x3 Numpy readable format. E.g. -
```
2759.48 0 1520.69 
0 2764.16 1006.81 
0 0 1
```
You may use the data from [OpenMVG benchmark repo](https://github.com/openMVG/SfM_quality_evaluation).

## Dependencies
OpenCV  
Numpy  
Matplotlib

## Usage
`python tinysfm.py dataset/Herz-Jesus-P8`  
It creates a 3D reconstruction from the images in the directory.

_Note: TinySFM assumes the photos in directory are named in the order in which they are captured as you move around the subject. Unordered names will affect the quality of reconstruction._


## License
MIT