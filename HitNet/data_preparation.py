# Some functions in this file are adapted from ai_badminton (Adobe Research)
# Original source: https://github.com/jhwang7628/monotrack/blob/main/modified-tracknet/train-hitnet.ipynb
# Licensed under the Adobe Research License - for non-commercial research use only.

import os
import sys
import cv2
import pandas as pd
import numpy as np
from skimage.transform import resize
from ai_badminton.pose import read_player_poses
from ai_badminton.trajectory import Trajectory

num_consec = 12 
left_window = 6
right_window = 0
speed = 1.0

COURT_OUTPUT = 'court_detection/court.txt'

def read_court(court_path):
    # Open the file and read the first four lines
    with open(court_path, "r") as file:
        lines = file.readlines()
        # Extract the first four lines (court corner coordinates)
    corners = []
    for i in range(4):  # Read only the first four lines
        x, y = map(float, lines[i].strip().split(";"))  # Convert to float
        corners.append((x, y))  # Store as (x, y) tuple
    return corners

def fetch_data(basedir: str, rally: str, for_train = False):
    '''
    Fetch the corners, trajectory, hits, poses from a rally

    Argument:
        basedir: the directiory path of clip
        rally: the name of rally, it's something like 'clip_1'
    
    Return:
        corners: the coordinate of 4 court corners
        trajectory: the processed trajectory
        hit: the label (0 = no hit, 1 = hit)
        bottom_player, top_player: poses data
    '''
    court_path = COURT_OUTPUT
    traj_path = f'{basedir}{rally}_ball.csv'  # E.g. videos/sample/clip_1/clip_1_ball.csv
    pose_path = f'{basedir}{rally}'           # E.g. videos/sample/clip_1/clip_1


    # Note: Consider using the same court for all rallies
    if not os.path.exists(court_path):
        print(f'{court_path} not found')
        return None
    court_pts = read_court(court_path)
    corners = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten()

    trajectory = Trajectory(traj_path, interp = False)
    poses = read_player_poses(pose_path)
    bottom_player, top_player = poses[0], poses[1]

    data_dict = {'corners': corners, 'trajectory': trajectory, 'bottom_player': bottom_player, 'top_player': top_player}

    if for_train:
        hit_path = f'{basedir}/hits/{rally}_hit.csv'
        hit = pd.read_csv(hit_path)
        hit = hit.values[:, 1]
        data_dict['hit'] = hit

    return data_dict

def resample(series, s):
    flatten = False
    if len(series.shape) == 1:
        series.resize((series.shape[0], 1))
        series = series.astype('float64')
        flatten = True
    series = resize(
        series, (int(s * series.shape[0]), series.shape[1]),
    )
    if flatten:
        series = series.flatten()
    return series  

def process(basedir: str, rally: str, for_train = False):
    '''
    Preparing data for a rally

    Argument: 
        basedir: the directiory path of match
        rally: the name of rally, it's something like 'clip_1'
        for_train: whether to prepare label for training stage
    
    Return:
        x_t: the processed features
        y_t: the processed binary label
    '''
    video_path = f'{basedir}{rally}.mp4'
    x_list, y_list = [], []

    # fetch data
    data_dict = fetch_data(basedir, rally, for_train)
    if data_dict is None:
        return None
    corners = data_dict['corners']
    trajectory = data_dict['trajectory']
    bottom_player = data_dict['bottom_player']
    top_player = data_dict['top_player']
    
    # open video by cv2
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    height, width = frame.shape[:2]      

    # resample data with speed = 1 (see utils/resample)
    trajectory.X = resample(np.array(trajectory.X), speed)
    trajectory.Y = resample(np.array(trajectory.Y), speed)
    bottom_player = resample(bottom_player.values, speed)
    top_player = resample(top_player.values, speed)

    min_len = min(len(trajectory.X), len(bottom_player), len(top_player))

    trajectory.X = trajectory.X[:min_len]
    trajectory.Y = trajectory.Y[:min_len]
    bottom_player = bottom_player[:min_len]
    top_player = top_player[:min_len]


    for i in range(num_consec):
        end = len(trajectory.X)-num_consec+i+1
        x_bird = np.array(list(zip(trajectory.X[i:end], trajectory.Y[i:end])))

        if x_bird is None or x_bird.size == 0:
            continue

        # Use entire pose
        x_pose = np.hstack([bottom_player[i:end], top_player[i:end]])

        # print(f"x_bird.shape = {x_bird.shape}")
        # print(f"x_pose.shape = {x_pose.shape}")
        # print(f"np.array([corners for j in range(i, end)]).shape = {np.array([corners for j in range(i, end)]).shape}")


        x = np.hstack([x_bird, x_pose, np.array([corners for j in range(i, end)])])
        x_list.append(x)

    # stack data for this rally
    if len(x_list) == 0:
        if not for_train:
            # 推論模式下給 dummy features
            feature_dim = 2 + bottom_player.shape[1] + top_player.shape[1] + corners.shape[0]
            dummy_x = np.zeros((1, feature_dim * num_consec))
            x_t = np.hstack([dummy_x])
            return x_t, None
        else:
            return None
    
    x_t = np.hstack(x_list)

    # Assume we only want predict binary outcome
    # Then no other processing is required
    if for_train:
        hit = data_dict['hit']
        hit = resample(hit, speed).round()
        y_new = np.array(hit)
        for i in range(num_consec):
            y = y_new[i:end]
            y_list.append(y)
        if right_window > 0:
            y_t = np.max(np.column_stack(y_list[left_window:-right_window]), axis=1)
        else:
            y_t = np.max(np.column_stack(y_list[left_window:]), axis=1)

        return x_t, y_t

    return x_t, None

def make_data_for_train(matches_dir, matches_lst):
    X_lst, y_lst = [], []
    for match in matches_lst:
        basedir = f'{matches_dir}/{match}'
        for video in os.listdir(f'{basedir}/rally_video/'):
            rally = video.split('.')[0]
            data = process(basedir, rally, for_train = True)
            if data is None:
                continue
            x, y = data
            X_lst.append(x)
            y_lst.append(y)

    X = np.vstack(X_lst)
    y = np.hstack(y_lst)
    
    return X, y

def make_data_for_predict(clip_dir, rally):
    X_lst = []
    data = process(clip_dir, rally, for_train = False)
    if data:
        x, _ = data
        X_lst.append(x)

    X = np.vstack(X_lst)
    
    return X