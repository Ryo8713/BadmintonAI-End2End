import numpy as np
import pandas as pd
from pathlib import Path

# 19 條骨架連線
COCO_BONES = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (7, 9), (6, 8),
    (8,10), (5, 6), (5,11), (6,12), (11,12),
    (11,13), (13,15), (12,14), (14,16),
]

def normalize_coords(coords, width=1280, height=720):
    coords = coords.astype(np.float64)
    coords[:, 0::2] /= width
    coords[:, 1::2] /= height
    return coords

def load_keypoints(csv_path):
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 1:].values    # 跳過 frame 欄
    data = normalize_coords(data)
    return data.reshape(len(df), -1, 2)  # (T, J, 2)

def extract_pos_center(kps):
    return np.mean(kps, axis=1)      # (T, 2)

def pad_to_length(data, T_max):
    T = data.shape[0]
    if T >= T_max:
        return data[:T_max]
    pad_shape = (T_max - T,) + data.shape[1:]
    return np.concatenate([data, np.zeros(pad_shape)], axis=0)

def compute_bones(kps, bones_idx):
    # 回傳 (T, B, 2)
    return np.stack([kps[:, j] - kps[:, i] for i, j in bones_idx], axis=1)

def process_clip(clip_name, tempose_root, T_max=100):
    """
    tempose_root\
      clip_1\
        clip_1_top.csv
        clip_1_bottom.csv
        clip_1_ball.csv
    """
    clip_dir = Path(tempose_root)

    top_csv    = clip_dir / f"{clip_name}_top.csv"
    bot_csv    = clip_dir / f"{clip_name}_bottom.csv"
    ball_csv   = clip_dir / f"{clip_name}_ball.csv"

    # 讀取並正規化 keypoints
    top    = load_keypoints(top_csv)
    bottom = load_keypoints(bot_csv)

    # 只取 X,Y (第 2,3 欄)，正規化
    ball_df = pd.read_csv(ball_csv)
    ball_xy = ball_df.iloc[:, 2:4].values / np.array([1280, 720])

    # 對齊最短長度並 padding
    n = min(len(top), len(bottom), len(ball_xy))
    top, bottom, ball_xy = top[:n], bottom[:n], ball_xy[:n]
    top    = pad_to_length(top, T_max)
    bottom = pad_to_length(bottom, T_max)
    ball_xy= pad_to_length(ball_xy, T_max)

    # 計算骨架向量
    t_top = compute_bones(top, COCO_BONES)
    t_bot = compute_bones(bottom, COCO_BONES)

    # 合併 joints + bones → (T, J+B, 2)
    top_all = np.concatenate([top, t_top], axis=1)
    bot_all = np.concatenate([bottom, t_bot], axis=1)

    # 最終 human_pose: (1, T, 2, J+B, 2)
    human_pose = np.stack([top_all, bot_all], axis=1)[None, ...]
    pos        = np.stack([extract_pos_center(top), extract_pos_center(bottom)], axis=1)[None, ...]
    shuttle    = ball_xy[None, ...]
    videos_len = np.array([n])
    labels     = np.array([0], dtype=int)

    # 儲存到 clip_dir
    np.save(clip_dir / 'JnB_bone.npy', human_pose)
    np.save(clip_dir / 'pos.npy', pos)
    np.save(clip_dir / 'shuttle.npy', shuttle)
    np.save(clip_dir / 'videos_len.npy', videos_len)
    np.save(clip_dir / 'labels.npy', labels)

    print(f"[✓] Saved .npy files into {clip_dir}")

if __name__ == "__main__":
    # 把下面路徑改成你本機的 tempose 根目錄
    tempose_root = r"C:\Badminton\BadmintonAI-End2End\tempose"
    process_clip("clip_1", tempose_root)
