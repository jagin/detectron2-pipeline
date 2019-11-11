import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer

from pipeline.pipeline import Pipeline
from pipeline.utils.colors import colors
from pipeline.utils.text import put_text


class AnnotateVideo(Pipeline):
    """Pipeline task for video annotation."""

    def __init__(self, dst, metadata_name, instance_mode=ColorMode.IMAGE):
        self.dst = dst
        self.metadata_name = metadata_name
        self.metadata = MetadataCatalog.get(self.metadata_name)
        self.instance_mode = instance_mode
        self.cpu_device = torch.device("cpu")
        self.video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        super().__init__()

    def map(self, data):
        dst_image = data["image"].copy()
        data[self.dst] = dst_image

        self.annotate_frame_num(data)
        self.annotate_predictions(data)

        return data

    def annotate_frame_num(self, data):
        dst_image = data[self.dst]
        frame_idx = data["frame_num"]

        put_text(dst_image, f"{frame_idx:04d}", (0, 0),
                 color=colors.get("white").to_bgr(),
                 bg_color=colors.get("black").to_bgr(),
                 org_pos="tl")

    def annotate_predictions(self, data):
        if "predictions" not in data:
            return

        dst_image = data[self.dst]
        dst_image = dst_image[:, :, ::-1]  # Convert OpenCV BGR to RGB format
        predictions = data["predictions"]

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_image = self.video_visualizer.draw_panoptic_seg_predictions(dst_image,
                                                                            panoptic_seg.to(self.cpu_device),
                                                                            segments_info)
        elif "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"].argmax(dim=0)
            vis_image = self.video_visualizer.draw_sem_seg(dst_image,
                                                           sem_seg.to(self.cpu_device))
        elif "instances" in predictions:
            instances = predictions["instances"]
            vis_image = self.video_visualizer.draw_instance_predictions(dst_image,
                                                                        instances.to(self.cpu_device))

        # Converts RGB format to OpenCV BGR format
        vis_image = cv2.cvtColor(vis_image.get_image(), cv2.COLOR_RGB2BGR)
        data[self.dst] = vis_image
