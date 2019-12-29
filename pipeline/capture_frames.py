from pipeline.capture_images import CaptureImages


class CaptureFrames(CaptureImages):
    """Pipeline task to capture image frames from directory."""

    def __init__(self, path, valid_exts=(".jpg", ".png")):
        self.path = path
        self.valid_exts = valid_exts
        self.frame_num = 0
        self.frame_count = -1

        super().__init__(path, valid_exts)

    def map(self, data):
        data["frame_num"] = self.frame_num
        self.frame_num += 1

        return data
