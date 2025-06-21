#!/usr/bin/env python
import os
import subprocess
import yaml
import torch
import pandas as pd
from pathlib import Path

from rally_clipping.get_result import timepoints_clipping
from TrackNetV3.predict import predict_traj
from TrackNetV3.denoise import smooth
from MMPose.detect_pose import process_pose
from HitNet.predict import predict as hitnet_detect

from TemPose.generate_npy import process_clip
from TemPose.predict_by_hit import per_hit_predict
from TemPose.TemPoseII import TemPoseII_TF

def main():
    # ——— 1. 配置路径 ——————————————————————————————————————————————
    video_path = Path('sample.mp4')
    name = video_path.stem
    RALLY_OUTPUT = Path('videos') / name

    COURT_DETECTION_PATH = 'court_detection/court-detection.exe'
    COURT_OUTPUT         = 'court_detection/court.txt'
    COURT_IMAGE          = 'court_detection/court_image.png'

    '''
    # ——— 2. Rally 切片 ——————————————————————————————————————————————
    print("\n[Message] Start rally clipping\n")
    timepoints_clipping(video_path)
    print("[Message] Rally clipping finished\n")

    # ——— 3. Court Detection ——————————————————————————————————————————
    print("\n[Message] Start court detection\n")
    subprocess.run(
        [COURT_DETECTION_PATH, str(video_path), COURT_OUTPUT, COURT_IMAGE, "10"],
        capture_output=True, text=True
    )
    print("[Message] Court detection finished\n")

    # ——— 4. 轨迹 & 姿态 预测 ————————————————————————————————————————
    print("\n[Message] Start trajectory & pose prediction\n")
    for clip in os.listdir(RALLY_OUTPUT):
        clip_dir  = RALLY_OUTPUT / clip
        clip_path = clip_dir / f"{clip}.mp4"

        traj_csv = predict_traj(clip_path, str(clip_dir))
        df       = pd.read_csv(traj_csv, encoding="utf-8")
        smooth(traj_csv, df)

        process_pose(clip_path, str(clip_dir), COURT_OUTPUT)
    print("[Message] Trajectory & pose prediction finished\n")

    # ——— 5. HitNet 击球检测 ————————————————————————————————————————
    print("\n[Message] Start hit detection\n")
    hitnet_detect(RALLY_OUTPUT)
    print("[Message] Hit detection finished\n")
    '''
    # ——— 6. TemPose 每击球分类 ——————————————————————————————————————
    # 6.1 载入模型配置 & 权重
    cfg = yaml.safe_load(Path("TemPose/config/config.yml").read_text())
    cfg_model = cfg["model"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemPoseII_TF(
        poses_numbers=cfg_model["input_dim"],
        time_steps=   cfg_model["sequence_length"],
        num_people=   cfg_model["num_people"],
        num_classes=  cfg_model["output_dim"],
        dim=          cfg_model["model_dim"],
        depth=        cfg_model["depth_t"],
        depth_int=    cfg_model["depth_n"],
        dim_head=     cfg_model["head_dim"],
        emb_dropout=  cfg["hyperparameters"]["dropout"],
    ).to(device)
    ck = torch.load("TemPose/model.pt", map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # 6.2 对每个 clip 生成 .npy 并对每次击球做预测
    for clip in os.listdir(RALLY_OUTPUT):
        clip_dir = RALLY_OUTPUT / clip
        print(f"\n[TemPose] Processing {clip} …")

        # ① 生成 npy 输入
        process_clip(clip, str(clip_dir), T_max=cfg_model["sequence_length"])

        # ② 读取击球时刻并逐一预测
        hit_csv = clip_dir / f"{clip}_hits.csv"
        if hit_csv.exists():
            per_hit_predict(model, clip_dir, hit_csv, cfg, device)
        else:
            print(f"[WARN] {hit_csv} not found, skipping per-hit prediction.")

    print("\n[All done]")

if __name__ == "__main__":
    main()
