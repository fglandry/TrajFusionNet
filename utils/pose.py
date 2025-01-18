import cv2 as cv
import math
from operator import itemgetter
import numpy as np
import torch

# from libs.simple_hrnet.SimpleHRNet import SimpleHRNet


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

MOST_STABLE_KEYPOINTS = [2, 5, 8, 9, 10, 11, 12, 13]
MOST_STABLE_EDGES = [(0,1), (0,2), (1,5), (2,5), (2,3), (5,6), (3,4), (6,7)]

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne,
#  Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]

COCO_BODY_PARTS = { "Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4,
                   "LShoulder": 5, "RShoulder": 6, "LElbow": 7, "RElbow": 8,
                   "LWrist": 9, "RWrist": 10, "LHip": 11, "RHip": 12,
                   "LKnee": 13, "RKnee": 14, "LAnkle": 15, "RAnkle": 16 }
COCO_MOST_STABLE_KEYPOINTS = [5, 6, 11, 12, 13, 14, 15, 16]
COCO_MOST_STABLE_EDGES = [(0,1), (1,3), (0,2), (2,3), (3,5), (2,4), (5,7), (4,6)]
COCO_V2_MOST_STABLE_KEYPOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
COCO_V2_MOST_STABLE_EDGES = [(0,1), (0,2), (1,2), (1,3), (3,5), 
                             (2,4), (4,6), (1,7), (2,8), (7,8),
                             (7,9), (8,10), (9,11), (10,12)]

HRNET_MODEL = None

def get_pose_keypoints(model_opts,
                       img,
                       use_openpose=False,
                       use_hrnet=True,
                       debug=False):
    
    keep_most_stable_keypoints = True
    add_distances_between_keypoints = False
    add_angles_between_limb_and_axis = False
    add_joints_angles = False
    add_joints_angles_diff = False
    if "pose_configs" in model_opts:
        if model_opts["pose_configs"].get("keep_most_stable_keypoints"):
            keep_most_stable_keypoints = True
        if model_opts["pose_configs"].get("add_distances_between_keypoints"):
            add_distances_between_keypoints = True
        if model_opts["pose_configs"].get("add_angles_between_limb_and_axis"):
            add_angles_between_limb_and_axis = True
        if model_opts["pose_configs"].get("add_joints_angles"):
            add_joints_angles = True
        if model_opts["pose_configs"].get("add_joints_angles_diff"):
            add_joints_angles_diff = True

    photo_height=img.shape[0]
    photo_width=img.shape[1]

    if use_openpose:
        most_stable_keypoints = MOST_STABLE_KEYPOINTS
        points = get_keypoints_using_openpose(
            img, photo_height, photo_width,
            weights="weights/graph_opt.pb",
            threshold=0)
    elif use_hrnet:
        most_stable_keypoints = COCO_V2_MOST_STABLE_KEYPOINTS # COCO_MOST_STABLE_KEYPOINTS
        points = get_keypoints_using_hrnet(
            img, photo_height, photo_width,
            weights="libs/simple_hrnet/weights/pose_hrnet_w48_384x288.pth",
            threshold=0)
    # Keep most stable points if set in config
    if keep_most_stable_keypoints:
        points = list(itemgetter(*most_stable_keypoints)(points))

    features = normalize_keypoints(points, photo_height, photo_width)

    if debug:
        for idx, point in enumerate(points):
            cv.ellipse(img, point, (1, 1), 0, 0, 
                       360, (0, 0, 255), cv.FILLED)
        
        edges = MOST_STABLE_EDGES if use_openpose else COCO_V2_MOST_STABLE_EDGES
        for edge in edges:
            cv.line(img, points[edge[0]], points[edge[1]], (0, 0, 255), 1)

    if add_distances_between_keypoints:
        features = calculate_distances_between_keypoints(features, points, photo_height, 
                                                         photo_width, use_hrnet=use_hrnet)
    if add_angles_between_limb_and_axis:
        features = calculate_angle_between_limb_and_axis(features, points,
                                                         use_hrnet=use_hrnet)
    if add_joints_angles:
        features = calculate_joints_angles(features, points)
    if add_joints_angles_diff:
        features = calculate_joints_angles_diff(features, points)

    return features

def get_keypoints_using_openpose(
        img, photo_height, photo_width,
        weights='weights/graph_opt.pb',
        threshold=0) -> list[tuple]:

    net = cv.dnn.readNetFromTensorflow(weights)
    net.setInput(cv.dnn.blobFromImage(img, 1.0, (photo_width, photo_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))

    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (photo_width * point[0]) / out.shape[3]
        y = (photo_height * point[1]) / out.shape[2]
        # Add a point if its confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > threshold else None)

    return points # [nb_keypoints, 2]

def get_keypoints_using_hrnet(
        img, photo_height, photo_width,
        weights="libs/simple_hrnet/weights/pose_hrnet_w48_384x288.pth",
        threshold=0) -> list[tuple]:

    global HRNET_MODEL
    if not HRNET_MODEL:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        HRNET_MODEL = SimpleHRNet(48, 17, weights,
                                  multiperson=False,
                                  device=device)
    out = HRNET_MODEL.predict(img)

    points = []

    for i in range(len(COCO_BODY_PARTS)):
        point = out[0][i]
        conf = point[2]
        x = point[1]
        y = point[0]
        # Add a point if its confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > threshold else None)

    return points # [nb_keypoints, 2]

def normalize_keypoints(points, photo_height, photo_width):
    norm_points = points.copy()
    for idx, point in enumerate(norm_points):
        try:
            norm_points[idx] = [(point[0] - photo_width/2) / (photo_width/2), \
                        (point[1] - photo_height/2) / (photo_height/2)]
        except:
            norm_points[idx] = [0, 0]
    # flatten list with x-values first followed by y-values
    flat_list_points = [sublist[0] for sublist in norm_points] + [sublist[1] for sublist in norm_points]
    # flat_list_points = [i for sublist in norm_points for i in sublist]
    return flat_list_points

def calculate_distances_between_keypoints(features, points, photo_height, photo_width,
                                          keep_most_stable=True, use_hrnet=False):
    max_dist = math.hypot(photo_height, photo_width)
    most_stable_edges = COCO_V2_MOST_STABLE_EDGES if use_hrnet else MOST_STABLE_EDGES
    if keep_most_stable:
        if use_hrnet:
            for edge in most_stable_edges:
                p1 = points[edge[0]]
                p2 = points[edge[1]]
                try:
                    features.append(math.dist(p1, p2)/max_dist)
                except:
                    features.append(0)
        else:
            raise Exception
    else:
        for idx1, p1 in enumerate(points):
            for idx2, p2 in enumerate(points):
                if idx1 == idx2:
                    continue
                try:
                    features.append(math.dist(p1, p2)/max_dist)
                except:
                    features.append(0)
    return features

def calculate_angle_between_limb_and_axis(features, points,
                                          keep_most_stable=True, use_hrnet=False):
    
    def _compute_angle(features, p1, p2):
        try:
            p1_np = np.array(p1)
            p2_np = np.array(p2)
            if np.array_equal(p1_np, p2_np):
                features.append(0)
                return features
            vector1 = np.subtract(p2_np, p1_np)
            vector1 = vector1 / np.linalg.norm(vector1)
            vector2 = np.array([1, 0]) # unit vector at 0 rad
            dot_product = np.dot(vector1, vector2)
            angle = np.arccos(dot_product)
            features.append(angle/(2*np.pi))
        except Exception as e:
            features.append(0)

    np.seterr(all='raise')
    most_stable_edges = COCO_V2_MOST_STABLE_EDGES if use_hrnet else MOST_STABLE_EDGES
    if keep_most_stable:
        if use_hrnet:
            for edge in most_stable_edges:
                p1 = points[edge[0]]
                p2 = points[edge[1]]
                _compute_angle(features, p1, p2)
        else:
            raise Exception
    else:
        for idx1, p1 in enumerate(points):
            for idx2, p2 in enumerate(points):
                if idx1 == idx2:
                    continue
                _compute_angle(features, p1, p2)
    return features
        
def calculate_joints_angles(features, points):
    np.seterr(all='raise')
    for idx1, p1 in enumerate(points):
        for idx2, p2 in enumerate(points):
            for idx3, p3 in enumerate(points):
                if idx1 == idx2 or idx2 == idx3 or idx1 == idx3:
                    continue
                try:
                    p1_np = np.array(p1)
                    p2_np = np.array(p2)
                    p3_np = np.array(p3)
                    if np.array_equal(p1_np, p2_np) or np.array_equal(p2_np, p3_np) or np.array_equal(p1_np, p3_np):
                        features.append(0)
                        continue
                    vector1 = np.subtract(p2_np, p1_np)
                    vector1 = vector1 / np.linalg.norm(vector1)
                    vector2 = np.subtract(p3_np, p2_np)
                    vector2 = vector2 / np.linalg.norm(vector2)
                    dot_product = np.dot(vector1, vector2)
                    angle = np.arccos(dot_product)
                    features.append(angle/(2*np.pi))
                except Exception as e:
                    features.append(0)
    return features

def calculate_joints_angles_diff(features, points):
    # TODO: implement
    return features
