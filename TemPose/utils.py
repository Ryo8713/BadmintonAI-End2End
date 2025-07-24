import numpy as np
import pandas as pd

def normalize_joints(
    arr: np.ndarray,  # shape: (T, J, 2)
    bbox: np.ndarray,  # shape: (T, 4)
    v_height=None,
    center_align=True,
):
    """
    Normalize joints over T frames for a single player.
    
    - arr: (T, J, 2)
    - bbox: (T, 4)
    - v_height: scalar or None
    - center_align: bool
    Return:
    - normalized_arr: (T, J, 2)
    """
    if v_height:
        dist = v_height / 4.0
    else:
        dist = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)  # (T, 1)

    arr_x = arr[:, :, 0]  # (T, J)
    arr_y = arr[:, :, 1]

    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / dist, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / dist, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2  # (T, 2)
        c_normalized = (center - bbox[:, :2]) / dist  # (T, 2)
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)  # (T, J, 2)

def normalize_position(arr: np.ndarray, court_info: dict):
    '''
    Normalized by court boundary.

    arr:    (T, 2, 2). There are N 'x' and N 'y'.
    Output: (T, 1, 2). Every 'x', 'y' in-court should be in [0, 1].
    '''
    # calculate the mean of arr
    arr = arr.mean(axis=1, keepdims=True)   # (T, 1, 2)

    x_mean = (court_info['border_L'] + court_info['border_R']) / 2
    y_mean = (court_info['border_U'] + court_info['border_D']) / 2
    x_dist = court_info['border_R'] - court_info['border_L']
    y_dist = court_info['border_D'] - court_info['border_U']

    x_normalized = (arr[:, :, 0] - x_mean) / x_dist + 0.5   # (T, 1)
    y_normalized = (arr[:, :, 1] - y_mean) / y_dist + 0.5   # (T, 1)
    return np.stack((x_normalized, y_normalized), axis=-1)  # (T, 1, 2)

def get_corner_camera(court_txt = 'court_detection/court.txt'):
    with open(court_txt, 'r') as f:
        lines = f.readlines()
    
    corners = []
    for i in range(4):
        parts = lines[i].strip().split(';')
        x = float(parts[0])
        y = float(parts[1])
        corners.append([x, y])
    
    corners = np.array(corners).T  # shape: (2, 4)
    return corners


def get_H(court_txt = 'court_detection/court.txt'):
    with open(court_txt, 'r') as f:
        lines = f.readlines()

    # find the line "Homography Matrix:"
    h_start = next(i for i, line in enumerate(lines) if "Homography Matrix:" in line)
    h_lines = lines[h_start + 1 : h_start + 4]

    H = []
    for line in h_lines:
        row = [float(x.strip()) for x in line.strip().replace('[', '').replace(']', '').replace(';', '').split(',')]
        H.append(row)

    return np.array(H)

def convert_homogeneous(arr: np.ndarray):
    '''
    The shape of 2D `arr` is (2, N). => The output will be (3, N).
    '''
    # print(arr.shape)
    return np.concatenate((arr, np.ones((1, arr.shape[1]))), axis=0)

def project(H: np.ndarray, P_prime: np.ndarray):
    '''
    Transform coordinates from the camera system to the court system.
    
    H: (3, 3)
    P_prime: (3, N)
    Output: (2, N)
    '''
    P = H @ P_prime
    P = P[:2, :] / P[-1, :]  # /= w
    return P

def get_court_info():
    '''
    Get the homography matrix and the 4 corners of the court in the court coordinate corresponding to the video.
    '''
    H = get_H()
    corner_camera = get_corner_camera()
    corner_camera = convert_homogeneous(corner_camera)

    corner_court = project(H, corner_camera)
    return {
        'H': H,
        'border_L': corner_court[0, 0],
        'border_R': corner_court[0, 1],
        'border_U': corner_court[1, 0],
        'border_D': corner_court[1, 2],
    }

def to_court_coordinate(
    arr_camera: np.ndarray,
    H
):
    '''
    Convert the camera coordinate system to the court coordinate system.
    
    Input:
        arr_camera: np.ndarray of shape (30, 2, 2)
            30 frames, each with 2 points (e.g., top/bottom), each point (x, y)
        H: homography matrix

    Output:
        np.ndarray of shape (30, 2, 2)
            Projected court coordinates
    '''
    court_coord = []
    for i in range(arr_camera.shape[0]):
        pts = arr_camera[i]                  # shape: (2, 2)
        pts_h = convert_homogeneous(pts)     # shape: (2, 3)
        pts_proj = project(H, pts_h)         # shape: (2, 2)
        court_coord.append(pts_proj)
    court_coord = np.stack(court_coord)      # shape: (30, 2, 2)
    return court_coord
