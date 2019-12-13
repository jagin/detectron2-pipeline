import cv2
import torch
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer

from pipeline.pipeline import Pipeline
from pipeline.utils.colors import colors
from pipeline.utils.text import put_text


class AnnotateVideo(Pipeline):
    """Pipeline task for video annotation."""

    def __init__(self, dst, metadata_name, instance_mode=ColorMode.IMAGE,
                 frame_num=True, predictions=True, pose_flows=True):
        self.dst = dst
        self.metadata_name = metadata_name
        self.metadata = MetadataCatalog.get(self.metadata_name)
        self.instance_mode = instance_mode
        self.frame_num = frame_num
        self.predictions = predictions
        self.pose_flows = pose_flows

        self.cpu_device = torch.device("cpu")
        self.video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        super().__init__()

    def map(self, data):
        dst_image = data["image"].copy()
        data[self.dst] = dst_image

        if self.frame_num:
            self.annotate_frame_num(data)
        if self.predictions:
            self.annotate_predictions(data)
        if self.pose_flows:
            self.annotate_pose_flows(data)

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

    def annotate_pose_flows(self, data):
        if "pose_flows" not in data:
            return

        predictions = data["predictions"]
        instances = predictions["instances"]
        keypoints = instances.pred_keypoints.cpu().numpy()
        l_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (6, 12), (5, 11), (11, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        dst_image = data[self.dst]
        height, width = dst_image.shape[:2]

        pose_flows = data["pose_flows"]
        pose_colors = list(colors.items())
        pose_colors_len = len(pose_colors)

        for idx, pose_flow in enumerate(pose_flows):
            pid = pose_flow["pid"]
            pose_color_idx = ((pid*10) % pose_colors_len + pose_colors_len) % pose_colors_len
            pose_color_bgr = pose_colors[pose_color_idx][1].to_bgr()
            (start_x, start_y, end_x, end_y) = pose_flow["box"].astype("int")
            cv2.rectangle(dst_image, (start_x, start_y), (end_x, end_y), pose_color_bgr, 2, cv2.LINE_AA)
            put_text(dst_image, f"{pid:d}", (start_x, start_y),
                     color=pose_color_bgr,
                     bg_color=colors.get("black").to_bgr(),
                     org_pos="tl")

            instance_keypoints = keypoints[idx]
            l_points = {}
            p_scores = {}
            # Draw keypoints
            for n in range(instance_keypoints.shape[0]):
                score = instance_keypoints[n, 2]
                if score <= 0.05:
                    continue
                cor_x = int(np.clip(instance_keypoints[n, 0], 0, width))
                cor_y = int(np.clip(instance_keypoints[n, 1], 0, height))
                l_points[n] = (cor_x, cor_y)
                p_scores[n] = score
                cv2.circle(dst_image, (cor_x, cor_y), 2, pose_color_bgr, -1)
            # Draw limbs
            for i, (start_p, end_p) in enumerate(l_pairs):
                if start_p in l_points and end_p in l_points:
                    start_xy = l_points[start_p]
                    end_xy = l_points[end_p]
                    start_score = p_scores[start_p]
                    end_score = p_scores[end_p]
                    cv2.line(dst_image, start_xy, end_xy, pose_color_bgr, int(2 * (start_score + end_score) + 1))
