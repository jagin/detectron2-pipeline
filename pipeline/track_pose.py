from pipeline.pipeline import Pipeline
from pipeline.libs.pose_tracker import PoseTracker


class TrackPose(Pipeline):

    def __init__(self, link_len=100, num=7, mag=30, match=0.2, orb_features=1000):
        self.tracker = PoseTracker(link_len=link_len, num=num, mag=mag, match=match,
                                   orb_features=orb_features)

        super().__init__()

    def map(self, data):
        self.track_pose(data)

        return data

    def track_pose(self, data):
        if "predictions" not in data:
            return

        predictions = data["predictions"]
        if "instances" not in predictions:
            return

        instances = predictions["instances"]
        if not instances.has("pred_keypoints"):
            return

        image = data["image"]
        keypoints = instances.pred_keypoints.cpu().numpy()
        scores = instances.scores
        num_instances = len(keypoints)
        assert len(scores) == num_instances

        data["pose_flows"] = self.tracker.track(image, keypoints, scores)

        return data
