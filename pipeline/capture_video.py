import cv2

from pipeline.pipeline import Pipeline
from pipeline.libs.file_video_capture import FileVideoCapture
from pipeline.libs.webcam_video_capture import WebcamVideoCapture


class CaptureVideo(Pipeline):
    """Pipeline task to capture video stream from file or webcam
    using faster, threaded method to reading video frames."""

    def __init__(self, src):
        if isinstance(src, int):
            self.cap = WebcamVideoCapture(src).start()
            self.frame_count = -1
        else:
            self.cap = FileVideoCapture(src).start()
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_num = 0

        super().__init__()

    def generator(self):
        """Yields the frame content and metadata."""

        while self.cap.running():
            image = self.cap.read()
            data = {
                "frame_num": self.frame_num,
                "image_id": f"{self.frame_num:06d}",
                "image": image,
            }
            self.frame_num += 1
            if self.filter(data):
                yield self.map(data)

    def cleanup(self):
        """Closes video file or capturing device.

        This function should be triggered after the pipeline completes.
        """

        self.cap.stop()
