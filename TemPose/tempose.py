import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms
import os
import glob
import pandas as pd
import pickle
import yaml

from TemPoseII import TemPoseII_TF
from DataUtils import PoseData_OL, RandomScaling, RandomFlip, RandomTranslation, select_trainingtest, filterOL, one_hot_ol
from data.dataset import get_stroke_types, get_bone_pairs, make_seq_len_same, create_bones, interpolate_joints, Dataset_npy_collated, prepare_npy_collated_loaders, prepare_npy_collated_one_side_loaders
from pathlib import Path
from train import train  # 如果有訓練階段可以使用

if __name__ == "__main__":
    # === 1. 讀取 config 設定 ===
    path = "./config/config.yml"
    model_path = "model.pt"

    with open(path) as f:
        config = yaml.safe_load(f)
        model_config = config["model"]

        match = config['dataset']['match']
        modeltype = model_config['model_name']
        time_steps = model_config['sequence_length']
        num_people = model_config['num_people']

        n_cls = model_config['output_dim']
        inp_dim = model_config['input_dim']
        d_t = model_config['depth_t']
        d_int = model_config['depth_n']
        d_l = model_config['model_dim']
        d_e = model_config['head_dim']
        drop = config['hyperparameters']['dropout']

    # === 2. 準備資料集 DataLoader ===
    loaders = prepare_npy_collated_loaders(
        Path("test"),
        pose_style="JnB_bone",
        num_workers=(num_people, num_people, num_people)
    )

    # === 3. 建立模型並移動到裝置 ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemPoseII_TF(
        poses_numbers=inp_dim,
        time_steps=time_steps,
        num_people=num_people,
        num_classes=n_cls,
        dim=d_l,
        depth=d_t,
        depth_int=d_int,
        dim_head=d_e,
        emb_dropout=drop
    )
    model.to(device)

    # === 4. 可選：載入已訓練模型參數 ===
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded trained model from {model_path}")

    # === 5. 在驗證集上推論 ===
    ans = []
    for X, pad, y in loaders[1]:
        pose, position, speed = X
        pose     = pose.to(device).float()
        position = position.to(device).float()
        speed    = speed.to(device).float()
        x   = pose.reshape(*pose.shape[:3], -1)
        pos = position.reshape(*position.shape[:2], -1)
        sp  = torch.cat([pos, speed], dim=-1)
        ans.extend(model.predict(x, sp, pad).numpy())

    # === 6. 輸出推論結果 ===
    #print("Predicted classes on validation set:", ans)

    # 儲存成文字檔
    with open("val_predicted_labels.txt", "w") as f:
        for label in ans:
            f.write(f"{label}\n")

    print("[INFO] Saved prediction results to val_predicted_labels.txt")
