##############################################################################
#
# Below code is based on
# https://github.com/YuliangXiu/PoseFlow
# --------------------------------------------------------
# PoseFlow: Efficient Online Pose Tracking (BMVC'18) (https://arxiv.org/abs/1802.00977)
# Credits: Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu
# --------------------------------------------------------

import numpy as np
import cv2
import math

import heapq
from munkres import Munkres

# get expand bbox surrounding single person's keypoints
def get_box(pose, frame_width, frame_height):
    xmin = np.min(pose[:, 0])
    xmax = np.max(pose[:, 0])
    ymin = np.min(pose[:, 1])
    ymax = np.max(pose[:, 1])

    return expand_bbox(xmin, xmax, ymin, ymax, frame_width, frame_height)


# expand bbox for containing more background
def expand_bbox(left, right, top, bottom, frame_width, frame_height):
    width = right - left
    height = bottom - top
    scale = 1.2
    ratio = (-width - height + math.sqrt(
        width ** 2 - 2 * width * height + 4 * width * height * scale + height ** 2)) / 2
    new_left = np.clip(left - ratio, 0, frame_width)
    new_right = np.clip(right + ratio, 0, frame_width)
    new_top = np.clip(top - ratio, 0, frame_height)
    new_bottom = np.clip(bottom + ratio, 0, frame_height)

    return [int(new_left), int(new_right), int(new_top), int(new_bottom)]


def flann_matching(orb_match1, orb_match2):
    kp1, des1 = orb_match1
    kp2, des2 = orb_match2

    # FLANN parameters
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=12,
                        key_size=12,
                        multi_probe_level=2)
    search_params = dict(checks=100)  # or pass empty dictionary
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann_matcher.knnMatch(des1, des2, k=2)

    cor = []
    # ratio test as per Lowe's paper
    for m_n in matches:
        if len(m_n) != 2:
            continue
        elif m_n[0].distance < 0.80 * m_n[1].distance:
            cor.append([kp1[m_n[0].queryIdx].pt[0], kp1[m_n[0].queryIdx].pt[1],
                        kp2[m_n[0].trainIdx].pt[0], kp2[m_n[0].trainIdx].pt[1],
                        m_n[0].distance])

    return np.array(cor)


def orb_matching(frame1, frame2, orb_features=10000):
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=orb_features, scoreType=cv2.ORB_FAST_SCORE)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    return flann_matching((kp1, des1), (kp2, des2))


# stack all already tracked people's info together(thanks @ZongweiZhou1)
def stack_all_pids(frame_tracks, last_pid, link_len):
    all_pids_info = []
    all_pids_fff = []  # boolean list, 'fff' means From Former Frame
    all_pids = [i for i in range(last_pid + 1)]
    frame_traks_len = len(frame_tracks)

    for idx in np.arange(frame_traks_len - 2, max(frame_traks_len - 2 - link_len, -1), -1):
        for pid in range(len(frame_tracks[idx][1])):
            if len(all_pids) == 0:
                return all_pids_info, all_pids_fff
            elif frame_tracks[idx][1][pid]['new_pid'] in all_pids:
                all_pids.remove(frame_tracks[idx][1][pid]['new_pid'])
                all_pids_info.append(frame_tracks[idx][1][pid])
                if idx == frame_traks_len - 2:
                    all_pids_fff.append(True)
                else:
                    all_pids_fff.append(False)

    return all_pids_info, all_pids_fff


# calculate number of matching points in one box from last frame
def find_region_cors_last(box_pos, all_cors):
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    x_min, x_max, y_min, y_max = box_pos
    x1_region_ids = set(np.where((x1 >= x_min) & (x1 <= x_max))[0].tolist())
    y1_region_ids = set(np.where((y1 >= y_min) & (y1 <= y_max))[0].tolist())
    region_ids = x1_region_ids & y1_region_ids

    return region_ids


# calculate number of matching points in one box from next frame
def find_region_cors_next(box_pos, all_cors):
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    x_min, x_max, y_min, y_max = box_pos
    x2_region_ids = set(np.where((x2 >= x_min) & (x2 <= x_max))[0].tolist())
    y2_region_ids = set(np.where((y2 >= y_min) & (y2 <= y_max))[0].tolist())
    region_ids = x2_region_ids & y2_region_ids

    return region_ids


# calculate IoU of two boxes(thanks @ZongweiZhou1)
def cal_bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])  # xmin
    yA = max(boxA[2], boxB[2])  # ymin
    xB = min(boxA[1], boxB[1])  # xmax
    yB = min(boxA[3], boxB[3])  # ymax

    if xA < xB and yA < yB:
        interArea = (xB - xA + 1) * (yB - yA + 1)
        boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
        boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea + 0.00001)
    else:
        iou = 0.0

    return iou


# calculate DeepMatching based Pose IoU(only consider top NUM matched keypoints)
def cal_pose_iou_dm(all_cors, pose1, pose2, num, mag):
    poses_iou = []
    for ids in range(len(pose1)):
        pose1_box = [pose1[ids][0] - mag, pose1[ids][0] + mag, pose1[ids][1] - mag, pose1[ids][1] + mag]
        pose2_box = [pose2[ids][0] - mag, pose2[ids][0] + mag, pose2[ids][1] - mag, pose2[ids][1] + mag]
        poses_iou.append(find_two_pose_box_iou(pose1_box, pose2_box, all_cors))

    return np.mean(heapq.nlargest(num, poses_iou))


# calculate general Pose IoU(only consider top NUM matched keypoints)
def cal_pose_iou(pose1_box, pose2_box, num, mag):
    pose_iou = []
    for row in range(len(pose1_box)):
        x1, y1 = pose1_box[row]
        x2, y2 = pose2_box[row]
        box1 = [x1 - mag, x1 + mag, y1 - mag, y1 + mag]
        box2 = [x2 - mag, x2 + mag, y2 - mag, y2 + mag]
        pose_iou.append(cal_bbox_iou(box1, box2))

    return np.mean(heapq.nlargest(num, pose_iou))


# calculate final matching grade
def cal_grade(l, w):
    return sum(np.array(l) * np.array(w))


# calculate DeepMatching Pose IoU given two boxes
def find_two_pose_box_iou(pose1_box, pose2_box, all_cors):
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]

    x_min, x_max, y_min, y_max = pose1_box
    x1_region_ids = set(np.where((x1 >= x_min) & (x1 <= x_max))[0].tolist())
    y1_region_ids = set(np.where((y1 >= y_min) & (y1 <= y_max))[0].tolist())
    region_ids1 = x1_region_ids & y1_region_ids

    x_min, x_max, y_min, y_max = pose2_box
    x2_region_ids = set(np.where((x2 >= x_min) & (x2 <= x_max))[0].tolist())
    y2_region_ids = set(np.where((y2 >= y_min) & (y2 <= y_max))[0].tolist())
    region_ids2 = x2_region_ids & y2_region_ids

    inter = region_ids1 & region_ids2
    union = region_ids1 | region_ids2
    pose_box_iou = len(inter) / (len(union) + 0.00001)

    return pose_box_iou


# hungarian matching algorithm (thanks @ZongweiZhou1)
def best_matching_hungarian(all_cors, all_pids_info, all_pids_fff, track_vid_next_fid,
                            weights, weights_fff, num, mag):
    box1_num = len(all_pids_info)
    box2_num = len(track_vid_next_fid[1])
    cost_matrix = np.zeros((box1_num, box2_num))

    for pid1 in range(box1_num):
        box1_pos = all_pids_info[pid1]['box_pos']
        box1_region_ids = find_region_cors_last(box1_pos, all_cors)
        box1_score = all_pids_info[pid1]['box_score']
        box1_pose = all_pids_info[pid1]['keypoints_pos']
        box1_fff = all_pids_fff[pid1]

        for pid2 in range(box2_num):
            box2_pos = track_vid_next_fid[1][pid2]['box_pos']
            box2_region_ids = find_region_cors_next(box2_pos, all_cors)
            box2_score = track_vid_next_fid[1][pid2]['box_score']
            box2_pose = track_vid_next_fid[1][pid2]['keypoints_pos']

            inter = box1_region_ids & box2_region_ids
            union = box1_region_ids | box2_region_ids
            dm_iou = len(inter) / (len(union) + 0.00001)
            box_iou = cal_bbox_iou(box1_pos, box2_pos)
            pose_iou_dm = cal_pose_iou_dm(all_cors, box1_pose, box2_pose, num, mag)
            pose_iou = cal_pose_iou(box1_pose, box2_pose, num, mag)
            if box1_fff:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights)
            else:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights_fff)

            cost_matrix[pid1, pid2] = grade

    m = Munkres()
    indexes = m.compute((-np.array(cost_matrix)).tolist())

    return indexes, cost_matrix
