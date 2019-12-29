# Detectron2 pipeline

Modular image processing pipeline using OpenCV and Python generators powered by [Detectron2](https://github.com/facebookresearch/detectron2).  

For detailed description how to construct image processing pipeline using OpenCV and Python generators
read the following Medium stories in order:
* [Modular image processing pipeline using OpenCV and Python generators](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696)
* [Video processing pipeline with OpenCV](https://medium.com/deepvisionguru/video-processing-pipeline-with-opencv-ac10187d75b)
* [How to embed Detectron2 in your computer vision project](https://medium.com/deepvisionguru/how-to-embed-detectron2-in-your-computer-vision-project-817f29149461)
* **[PoseFlow - real-time pose tracking](https://medium.com/deepvisionguru/poseflow-real-time-pose-tracking-7f8062a7c996)**

## Setup environment

This project is using [Conda](https://conda.io) for project environment management.

Setup the project environment:

    $ conda env create -f environment.yml
    $ conda activate detectron2-pipeline
    
or update the environment if you `git pull` the repo previously:

    $ conda env update -f environment.yml
    
Install Detectron2 in a folder one level above the `detectron2-pipeline` project folder

    $ cd ..
    $ git clone https://github.com/facebookresearch/detectron2.git
    $ cd detectron2
    $ git checkout 3def12bdeaacd35c6f7b3b6c0097b7bc31f31ba4
    $ python setup.py build develop
    
We checkout `3def12bdeaacd35c6f7b3b6c0097b7bc31f31ba4` commit to ensure that you can use the code
out of the box with this repo.

If you got any problems with Detectron2 installation please refer to
[INSTALL.md](https://github.com/facebookresearch/detectron2/blob/3def12bdeaacd35c6f7b3b6c0097b7bc31f31ba4/INSTALL.md).

## Demo

Run the command to execute inferences on images from the selected folder:

    $ python process_images.py -i assets/images/friends -p

By default the instance segmentation model is used from `configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`.
You can try other models with `--config-file` option, for example:

    $ python process_images.py -i assets/images/friends -p --config-file configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml

For video processing run:

    $ python process_video.py -i assets/videos/walk.small.mp4 -p -d -ov walk.small.mp4

## Tests

`pytest` is used as a test framework. All tests are stored in `tests` folder. Run the tests:

```bash
$ pytest
```

## Resources and Credits

* For Unix like pipeline idea credits goes to this [Gist](https://gist.github.com/alexmacedo/1552724)
* The source of the example images and videos is [pixbay](https://pixabay.com)
* Some ideas and code snippets are borrowed from [pyimagesearch](https://www.pyimagesearch.com/)
* Color constants from [Python Color Constants Module](https://www.webucator.com/blog/2015/03/python-color-constants-module/)
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [Faster video file FPS with cv2.VideoCapture and OpenCV](https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/)
* [Applications of Foreground-Background separation with Semantic Segmentation](https://www.learnopencv.com/applications-of-foreground-background-separation-with-semantic-segmentation/)
* Pose tracking based on [PoseFlow: Efficient Online Pose Tracking](https://github.com/YuliangXiu/PoseFlow)

## License

[MIT License](LICENSE)