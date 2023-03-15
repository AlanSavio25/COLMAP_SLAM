# COLMAP-SLAM
[3D Vision](https://www.cvg.ethz.ch/teaching/3dvision/) project supervised by [Paul-Edouard](https://github.com/Skydes)


## Abstract
State-of-the-art Structure from Motion algorithms such as [COLMAP](https://github.com/colmap/colmap) are highly robust in reconstruction but are slow and often don't scale well. This makes them unsuitable for long video data. On the other hand, SLAM systems can process videos (sequential images) in real-time but fall behind COLMAP in map quality. The goal of this project is to combine the best of both worlds to obtain a fast, robust and scalable SLAM system. We demonstrate that we partially achieved our goals by utilizing components of COLMAP and ideas from [ORB-SLAM](https://github.com/raulmur/ORB_SLAM). The quality of our map, however, is not yet comparable with the state-of-the-art.

## Installation
We strongly recommend using Linux. It has been tested out on Ubuntu 20.04.1 LTS.

1. Install [COLMAP](https://colmap.github.io/)
2. Install [pycolmap](https://github.com/colmap/pycolmap)
3. Install [hloc](https://github.com/cvg/Hierarchical-Localization)
4. Install [pyceres](https://github.com/cvg/pyceres)
5. Install [Open3D](http://www.open3d.org/)
6. Install [OpenCV](https://pypi.org/project/opencv-python/)

If running on WSL, you need to make sure you have OpenGL installed and can launch graphical windows for Open3D to work. If there are issues running, try setting the following options:

```
export LIBGL_ALWAYS_INDIRECT=0
export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450
export LIBGL_ALWAYS_SOFTWARE=1
```

## APP
To run the app, run `python app.py`, this will will launch two windows, a frame viewer and the reconstruction window. On the right side of the main reconstruction window is settings side bar. 

### Loading a reconstruction
At the top of the side bar is an option to load one of the existing reconstructions from the `out/` folder. Simply click `Load Rec` and selection one of the `.bin` files in a folder to reload the points, path, and cameras.

### Running a reconstruction
To run the app, ensure the correct data path is shown in the side bar. The data path should be the parent folder of a ordered frames from a video. Adjust the following the settings then hit "Run".
- Frames to skip - Number of frames to skip in the source data set, useful to speed up operation when a high frame rate video is processed
- Max number of frames - The maximum number of frames to process
- Max frames for initialization - The number of frames to use for the initialization step, the larger this value, the more likely a robust structure is generated, however it it computationally intensive
- Optical flow threshold - The required optical flow between the last keyframe and the current image for it to become a keyframe
- Feature extractor - The feature extractor to generate points of interest (SuperPoint, ORB)
- Feature matcher - The matching algorithm to align points across from frames (SuperGlue, ORB Hamming/Nearest Neighbor)

If the per-frame callback is enabled (not running with the `-f` flag), then the reconstruction and video windows will update each time a new frame is processed. The video window will show the current frame and the last keyframe, along with the tracked points and their optical flow. It will also output the summary of the reconstruction at this point in time on the right. The reconstruction window will then show a live view of the cameras and points as they get registered.

### Visualization settings
Once a reconstruction is done, visualization settings can be adjusted to better analyze the data. There are also a few tools to help gain insight such as the camera track slider and the start and end image sliders.

Camera track slider will allow you to select a 3D point's id and draw a line to all the cameras that it is associated with.

The start and end image sliders will clip the displayed data to only the points and cameras that exist between the selected start and end frame ids.

### Options
Running `python app.py -h` will print out the help text for the following 2 options.

To immediately launch the application with a specific data directory, run `python app.py -d ./path/to/images` or `python app.py --dir ./path/to/images`.

If you are having issues with OpenGL causing a segmentation fault in WSL or an older system (known bug of Open3D), you can launch the app without the per-frame callback which should fix the issue. Run `python app.py -f` or `python app.py --fast`.

## General Pipeline

![pipeline](https://user-images.githubusercontent.com/17593719/173319164-a413e146-b05c-4326-a851-a81927f35f8a.png)
