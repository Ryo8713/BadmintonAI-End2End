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
    hp  = torch.from_numpy(np.load(clip_dir/"JnB_bone.npy")).float().to(device)   # (1, T, 2, J+B, 2)
    pos = torch.from_numpy(np.load(clip_dir/"pos.npy")).float().to(device)        # (1, T, 2, 2)
    sh  = torch.from_numpy(np.load(clip_dir/"shuttle.npy")).float().to(device)    # (1, T, 2)

    df_hits = pd.read_csv(hit_csv)
    hit_frames = df_hits.loc[df_hits['hit']==1, 'frame'].astype(int).tolist()

    T = cfg["model"]["sequence_length"]
    stroke_names = get_stroke_types()  # 0..34 的名稱列表

    print(f"Found {len(hit_frames)} hits, window size {T}")

    events = []

    for i, hf in enumerate(hit_frames):
        seg_hp  = slice_and_pad(hp[0].cpu().numpy(),  hf, T)[None]   # (1, T, 2, J+B, 2)
        seg_pos = slice_and_pad(pos[0].cpu().numpy(), hf, T)[None]   # (1, T, 2, 2)
        seg_sh  = slice_and_pad(sh[0].cpu().numpy(),  hf, T)[None]   # (1, T, 2)

        valid_len = min(T, hp.shape[1] - max(0, hf - T//2))
        t_pad = torch.tensor([valid_len], device=device)

        x = torch.tensor(seg_hp,  dtype=torch.float32, device=device)
        p = torch.tensor(seg_pos, dtype=torch.float32, device=device)
        s = torch.tensor(seg_sh,  dtype=torch.float32, device=device)

        b, t, m, d = p.shape  # (1, 30, 2, 2)
        p = p.view(b, t, m * d)  # (1, 30, 4)

        # print("X: ", x.shape)   # (1, 30, 2, 36, 2)
        # print("P: ", p.shape)   # (1, 30, 4)
        # print("S: ", s.shape)   # (1, 30, 2)

        sp = torch.cat([s, p], dim=-1)  # (b, t, 6)

        b,t,m,j,d = x.shape
        x = x.view(b, t, m, j*d)
        sp = sp.view(b, t, -1)
        
        with torch.no_grad():
            pred_idx = model.predict(x, sp, t_pad)
        name = stroke_names[pred_idx.item()]
        events.append((hf, name))

    return events

