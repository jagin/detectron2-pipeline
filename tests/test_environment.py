import cv2

def test_opencv_version():
    assert cv2.__version__ >= '4.0'