from rally_clipping.get_result import timepoints_clipping
from TrackNetV3.predict import predict_traj
from TrackNetV3.denoise import smooth
from MMPose.detect_pose import process_pose
from HitNet.predict import predict
from pathlib import Path
from tqdm import tqdm
import subprocess
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':

    video_path = Path('sample.mp4')
    name = video_path.stem

    RALLY_OUTPUT = f'videos/{name}/'
    COURT_DETECTION_PATH = 'court_detection/court-detection.exe'
    COURT_OUTPUT = 'court_detection/court.txt'
    COURT_IMAGE = 'court_detection/court_image.png'
    CLIP_INFO_DIR = 'clips/'

    # rally clipping
    print("\n[Message] Start rally clipping\n")
    timepoints_clipping(video_path)
    print("\n[Message] Rally clipping finished\n")

    # court detection
    print("\n[Message] Start court detection\n")
    result = subprocess.run(
        [COURT_DETECTION_PATH, video_path, COURT_OUTPUT, COURT_IMAGE, "10"],
        capture_output=True, text=True
    )
    print("[Message] The court information is stored in court_detection/")
    print("\n[Message] Court detection finished\n")

    # TrackNet & mmpose
    print("\n[Message] Start trajectory & pose prediction\n")
    for clip in os.listdir(RALLY_OUTPUT):
        output_dir = f'{RALLY_OUTPUT}{clip}/'
        clip_path = Path(f'{output_dir}{clip}.mp4')
        traj_csv_file = predict_traj(clip_path, output_dir)
        df = pd.read_csv(traj_csv_file, encoding="utf-8")
        smooth(traj_csv_file, df)
        process_pose(clip_path, output_dir, COURT_OUTPUT)
    print("[Message] Predictions stored in videos/")
    print("\n[Message] Trajectory & pose prediction finished\n")

    # HitNet
    print("\n[Message] Start hit detection\n")
    predict(RALLY_OUTPUT)
    print("[Message] Predictions stored in videos/")
    print("\n[Message] Hit detection finished\n")
