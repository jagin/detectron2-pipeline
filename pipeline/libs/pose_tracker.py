import numpy as np
import cv2

import pipeline.utils.pose_flow as pf


class PoseTracker:
    def __init__(self, link_len=100, num=7, mag=30, match=0.2, orb_features=1000):
        self.frame_tracks = []
        self.last_pid = 0
        self.link_len = link_len
        self.num = num
        self.mag = mag
        self.match = match
        self.orb_features = orb_features
        self.orb = cv2.ORB_create(nfeatures=orb_features, scoreType=cv2.ORB_FAST_SCORE)

    def track(self, frame, keypoints, scores):
        frame_h, frame_w = frame.shape[:2]
        pe_tracks = []
        pose_pids = []
        weights = [1, 2, 1, 2, 0, 0]
        weights_fff = [0, 1, 0, 1, 0, 0]

        if len(keypoints) == 0:
            return pose_pids

        # Based on PoseFlow (https://github.com/YuliangXiu/PoseFlow)
        for (i, instane_keypoints) in enumerate(keypoints):
            pe_track = {}
            pe_track["box_pos"] = pf.get_box(instane_keypoints, frame_w, frame_h)
            pe_track["box_score"] = scores[i]
            pe_track["keypoints_pos"] = instane_keypoints[:, 0:2]
            pe_track["keypoints_score"] = instane_keypoints[:, -1]

            # init tracking info of the first frame
            if len(self.frame_tracks) == 0:
                pe_track["new_pid"] = i
                pe_track["match_score"] = 0
                self.last_pid = i
            pe_tracks.append(pe_track)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # find the keypoints and descriptors with ORB
        frame_kp, frame_desc = self.orb.detectAndCompute(frame, None)
        self.frame_tracks.append((frame, pe_tracks, frame_kp, frame_desc))

        if len(self.frame_tracks) == 1:  # Is it the first frame?
            for i in range(len(keypoints)):
                (x1, x2, y1, y2) = pe_tracks[i]["box_pos"]
                pose_pids.append({
                    "pid": pe_tracks[i]["new_pid"],
                    "score": pe_tracks[i]["match_score"],
                    "box": np.array([x1, y1, x2, y2])
                })
            return pose_pids

        # Get previous frame data
        prev_frame, prev_pe_track, prev_frame_kp, prev_frame_desc = self.frame_tracks[-2]

        # Match ORB descriptor vectors for current and previous frame
        # with a FLANN (Fast Library for Approximate Nearest Neighbors) based matcher
        all_cors = pf.flann_matching((prev_frame_kp, prev_frame_desc), (frame_kp, frame_desc))
        # Stack all already tracked people's info together
        curr_all_pids, curr_all_pids_fff = pf.stack_all_pids(self.frame_tracks, self.last_pid, self.link_len)
        # Hungarian matching algorithm
        match_indexes, match_scores = pf.best_matching_hungarian(
            all_cors, curr_all_pids, curr_all_pids_fff, self.frame_tracks[-1], weights, weights_fff, self.num, self.mag)

        for pid1, pid2 in match_indexes:
            if match_scores[pid1][pid2] > self.match:
                self.frame_tracks[-1][1][pid2]["new_pid"] = curr_all_pids[pid1]["new_pid"]
                self.last_pid = max(self.last_pid, self.frame_tracks[-1][1][pid2]["new_pid"])
                self.frame_tracks[-1][1][pid2]["match_score"] = match_scores[pid1][pid2]

        # add the untracked new person
        for next_pid in range(len(self.frame_tracks[-1][1])):
            if "new_pid" not in self.frame_tracks[-1][1][next_pid]:
                self.last_pid += 1
                self.frame_tracks[-1][1][next_pid]["new_pid"] = self.last_pid
                self.frame_tracks[-1][1][next_pid]["match_score"] = 0

        for i in range(len(self.frame_tracks[-1][1])):
            (x1, x2, y1, y2) = self.frame_tracks[-1][1][i]["box_pos"]
            pose_pids.append({
                "pid": self.frame_tracks[-1][1][i]["new_pid"],
                "score": self.frame_tracks[-1][1][i]["match_score"],
                "box": np.array([x1, y1, x2, y2])
            })

        return pose_pids
