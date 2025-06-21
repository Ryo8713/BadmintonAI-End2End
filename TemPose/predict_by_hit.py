# predict_by_hit.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from TemPose.TemPoseII import TemPoseII_TF
from TemPose.data.dataset import create_bones, interpolate_joints, get_stroke_types  # 如果需要

def slice_and_pad(arr, center, T):
    half = T//2
    start = max(0, center-half)
    end   = min(arr.shape[0], center+half)
    seg   = arr[start:end]
    if seg.shape[0] < T:
        pad_shape = (T-seg.shape[0],) + seg.shape[1:]
        seg = np.concatenate([seg, np.zeros(pad_shape, dtype=seg.dtype)], axis=0)
    return seg

def per_hit_predict(model, clip_dir: Path, hit_csv: Path, cfg: dict, device: torch.device):
    """
    model: 已加载好权重并 eval() 的 TemPoseII_TF
    clip_dir: 包含 clip_1_top.csv, clip_1_bottom.csv, clip_1_ball.csv, clip_1_hits.csv 的文件夹
    hit_csv:    clip_1_hits.csv，每行一个 frame, hit=1 表示有击球帧
    cfg:        从 config.yml 里读出的 model config 字典
    """
    # 1) 先把刚才生成的 npy 读进来
    hp = torch.from_numpy(np.load(clip_dir/"JnB_bone.npy")).float().to(device)   # (1, T, 2, J+B, 2)
    pos= torch.from_numpy(np.load(clip_dir/"pos.npy"   )).float().to(device)   # (1, T, 2, 2)
    sh = torch.from_numpy(np.load(clip_dir/"shuttle.npy")).float().to(device)   # (1, T, 2)
    # videos_len, labels 通常不需要
    df_hits = pd.read_csv(hit_csv)
    # 假设它有一列叫 'hit'，和一列 'frame'，我们取所有 hit==1 的帧号
    hit_frames = df_hits.loc[df_hits['hit']==1, 'frame'].astype(int).tolist()

    T = cfg["model"]["sequence_length"]
    stroke_names = get_stroke_types()  # 0..34 的名称列表

    print(f"Found {len(hit_frames)} hits, window size {T}")

    for i, hf in enumerate(hit_frames):
        # slice_and_pad: 从 hf 向前/后截 T 帧并 pad 到固定长度
        seg_hp  = slice_and_pad(hp[0].cpu().numpy(),  hf, T)[None]   # (1, T, 2, J+B, 2)
        seg_pos = slice_and_pad(pos[0].cpu().numpy(), hf, T)[None]  # (1, T, 2, 2)
        seg_sh  = slice_and_pad(sh[0].cpu().numpy(),  hf, T)[None]  # (1, T, 2)

        x = torch.tensor(seg_hp,  dtype=torch.float32, device=device)
        p = torch.tensor(seg_pos, dtype=torch.float32, device=device)
        s = torch.tensor(seg_sh,  dtype=torch.float32, device=device)

        # reshape 成模型需要的形状
        b,t,m,j,d = x.shape
        x = x.view(b, t, m, j*d)
        p = p.view(b, t, -1)

        with torch.no_grad():
            pred_idx = model.predict(x, p, torch.tensor([hf], device=device))
        print(f"Hit {i:02d} @frame {hf:04d}: → [{pred_idx.item()}] {stroke_names[pred_idx.item()]}")
