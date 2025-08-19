from HitNet.data_preparation import make_data_for_predict
from HitNet.hitnet import HitNet
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

num_consec = 7 
dim = 70
batch_size = 4096
threshold = 0.56

MODEL_PATH = 'HitNet/hitnet_recall.pth'

def predict(rally_output_dir, start_frame, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HitNet(dim, num_consec=num_consec)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()

    for clip in os.listdir(rally_output_dir):
        if not clip.startswith("clip_"):
            continue
        clip_dir = f'{rally_output_dir}/{clip}/'
        # print(f"\n[HitNet] Processing {clip} ...")
        X = make_data_for_predict(clip_dir, clip)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) >= threshold).int()
                all_preds.extend(preds.cpu().numpy())

        hits = pd.DataFrame(all_preds, columns=['hit'])
        hits['frame'] = start_frame[clip] + hits.index  # 把 index 存進欄位
        hits = hits[['frame', 'hit']]  # 調整欄位順序
        hits.to_csv(f'{clip_dir}/{clip}_hits.csv', index=False)