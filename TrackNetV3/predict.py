import os
import cv2

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from pathlib import Path
from TrackNetV3.utils import *

def load_tracknet_model():
    model_file = 'TrackNetV3/exp/model_best.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_file, weights_only=True, map_location=device)
    param_dict = checkpoint['param_dict']
    model = get_model(param_dict['model_name'], param_dict['num_frame'], param_dict['input_type']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def predict_traj(video_file: Path, save_dir: str, model, verbose = False, draw = False):

    num_frame = 3
    batch_size = 4

    video_name = video_file.stem
    video_format = video_file.suffix.lstrip('.')  # 'mp4'
    out_video_file = f'{save_dir}/{video_name}_pred.{video_format}'
    out_csv_file = f'{save_dir}/{video_name}_ball.csv'

    print("Video name:", video_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Video output configuration
    if video_format == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    elif video_format == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError('Invalid video format.')

    # Write csv file head
    f = open(out_csv_file, 'w')
    f.write('Frame,Visibility,X,Y\n')

    # Cap configuration
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success = True
    frame_count = 0
    num_final_frame = 0
    ratio = h / HEIGHT

    if draw:
        out = cv2.VideoWriter(out_video_file, fourcc, fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frame:
        print(f"[Skip] {video_name} has only {total_frames} frames (need at least {num_frame})")
        success = False

        # fill 0 for missing frames
        for i in range(total_frames):
            f.write('0,0,0,0\n')

    with tqdm(total=total_frames, unit='frames', desc='Processing Video') as pbar:
        while success:
            if verbose:
                print(f'Number of sampled frames: {frame_count}')
            # Sample frames to form input sequence
            frame_queue = []
            for _ in range(num_frame*batch_size):
                success, frame = cap.read()
                if not success:
                    break
                else:
                    frame_count += 1
                    frame_queue.append(frame)
                    pbar.update(1)

            if not frame_queue:
                break
            
            # If mini batch incomplete
            if len(frame_queue) % num_frame != 0:
                frame_queue = []
                # Record the length of remain frames
                num_final_frame = len(frame_queue) +1

                if verbose:
                    print(num_final_frame)

                # Adjust the sample timestampe of cap
                frame_count = frame_count - num_frame*batch_size
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                # Re-sample mini batch
                for _ in range(num_frame*batch_size):
                    success, frame = cap.read()
                    if not success:
                        break
                    else:
                        frame_count += 1
                        frame_queue.append(frame)
                        pbar.update(1)
                if len(frame_queue) % num_frame != 0:
                    continue

            if len(frame_queue) < num_frame:
                if verbose:
                    print(f"[Warning] Frame queue too short ({len(frame_queue)}), skipping...")
                break
            
            x = get_frame_unit(frame_queue, num_frame)
            
            # Inference
            with torch.no_grad():
                y_pred = model(x.cuda())
            y_pred = y_pred.detach().cpu().numpy()
            h_pred = y_pred > 0.5
            h_pred = h_pred * 255.
            h_pred = h_pred.astype('uint8')
            h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)
            
            for i in range(h_pred.shape[0]):
                if num_final_frame > 0 and i < (num_frame*batch_size - num_final_frame-1):
                    if verbose:
                        print('aaa')
                    # Special case of last incomplete mini batch
                    # Igore the frame which is already written to the output video
                    continue 
                else:
                    img = frame_queue[i].copy()
                    cx_pred, cy_pred = get_object_center(h_pred[i])
                    cx_pred, cy_pred = int(ratio*cx_pred), int(ratio*cy_pred)
                    vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
                    # Write prediction result
                    f.write(f'{frame_count-(num_frame*batch_size)+i},{vis},{cx_pred},{cy_pred}\n')
                    # print(frame_count-(num_frame*batch_size)+i)
                    if draw:
                        if cx_pred != 0 or cy_pred != 0:
                            cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                        out.write(img)

    cap.release()
    if draw:
        out.release()
    print(f'[TrackNet] {video_name} done.')

    return out_csv_file

    '''
    Show trajectory & Event detection not pipe
    '''