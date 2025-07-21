#!/usr/bin/env python
import os
import subprocess
import yaml
import torch
import pandas as pd
from pathlib import Path
import cv2
from collections import Counter
from mmpose.apis import MMPoseInferencer

from rally_clipping.get_result import timepoints_clipping
from TrackNetV3.predict       import predict_traj, load_tracknet_model
from TrackNetV3.denoise       import smooth
from MMPose.detect_pose        import process_pose
from HitNet.predict           import predict as hitnet_detect

from TemPose.generate_npy     import process_clip
from TemPose.predict_by_hit   import per_hit_predict, slice_and_pad, get_stroke_types
from TemPose.TemPoseII        import TemPoseII_TF

from team_classifier.sport_player_team_classifier import predict_teams, train_yolo

from utils import frame_to_timestamp, visualize_hits_in_video, recording_execution_time, print_execution_time_with_plot

def main():
    # ——— 0. Paths & config —————————————————————————————————————
    video_path = Path('match8.mp4')
    name       = video_path.stem
    RALLY_OUTPUT_DIR = Path('videos') / name
    RESULT_OUTPUT_DIR = Path('results') / name

    COURT_DET_EXE = 'court_detection/court-detection.exe'
    COURT_OUTPUT  = 'court_detection/court.txt'
    COURT_IMAGE   = 'court_detection/court_image.png'

    draw = False # True if you want to visualize the predictions
    logs = {}    # for recording execution time

    # mkdir RESULT_OUTPUT_DIR
    os.makedirs(RESULT_OUTPUT_DIR, exist_ok = True)
    '''
    # ——— 1. Rally clipping ——————————————————————————————————————————————
    print("\n[Message] Start rally clipping\n")
    recording_execution_time(logs, "Start Rally Clipping")
    timepoints_clipping(video_path)
    print("[Message] Rally clipping finished\n")
    recording_execution_time(logs, "End Rally Clipping")
    
    # ——— 2. Court Detection ——————————————————————————————————————————
    print("\n[Message] Start court detection\n")
    recording_execution_time(logs, "Start Court Detection")
    subprocess.run(
        [COURT_DET_EXE, str(video_path), COURT_OUTPUT, COURT_IMAGE, "400"], 
        capture_output=True, text=True
    )
    print("[Message] Court detection finished\n")
    recording_execution_time(logs, "End Court Detection")
    '''
    # ——— 3. Trajectory & Pose Prediction —————————————————————————————————————
    print("\n[Message] Start trajectory & pose prediction\n")
    recording_execution_time(logs, "Start Trajectory & Pose Prediction")
    track_model = load_tracknet_model()
    inferencer = MMPoseInferencer('human', device='cuda')
    for clip in os.listdir(RALLY_OUTPUT_DIR):
        clip_dir  = RALLY_OUTPUT_DIR / clip
        clip_path = clip_dir / f"{clip}.mp4"

        traj_csv = predict_traj(clip_path, str(clip_dir), track_model, verbose=False, draw=draw)
        # df       = pd.read_csv(traj_csv, encoding="utf-8")
        # smooth(traj_csv, df)

        process_pose(inferencer, clip_path, str(clip_dir), COURT_OUTPUT, draw)
    print("[Message] Trajectory & pose prediction finished\n")
    recording_execution_time(logs, "End Trajectory & Pose Prediction")
    
    # ——— 4. HitNet ————————————————————————————————————————
    print("\n[Message] Start hit detection\n")
    recording_execution_time(logs, "Start Hit Detection")
    hitnet_detect(RALLY_OUTPUT_DIR)
    print("[Message] Hit detection finished\n")
    recording_execution_time(logs, "End Hit Detection")
    
    # ——— 5. Team Classification ——————————————————————————————————————
    print("\n[Message] Start team classification\n")
    recording_execution_time(logs, "Start Team Classification")
    classifier = train_yolo(video_path)
    for clip in os.listdir(RALLY_OUTPUT_DIR):
        clip_dir = RALLY_OUTPUT_DIR / clip
        print(f"\n[Team] Processing {clip} …")
        predict_teams(clip_dir, clip, classifier, draw)
    print("[Message] Team classification finished\n")
    recording_execution_time(logs, "End Team Classification")
    
    # ——— 6. TemPose  ——————————————————————————————————————
    # load model config
    print("\n[Message] Start TemPose\n")
    recording_execution_time(logs, "Start TemPose")
    cfg = yaml.safe_load(Path("TemPose/config/config.yml").read_text())
    mcfg = cfg["model"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemPoseII_TF(
        poses_numbers=mcfg["input_dim"],
        time_steps=   mcfg["sequence_length"],
        num_people=   mcfg["num_people"],
        num_classes=  mcfg["output_dim"],
        dim=          mcfg["model_dim"],
        depth=        mcfg["depth_t"],
        depth_int=    mcfg["depth_n"],
        dim_head=     mcfg["head_dim"],
        emb_dropout=  cfg["hyperparameters"]["dropout"],
    ).to(device)
    ck = torch.load("TemPose/model.pt", map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # Prepare containers for summarization
    overall_counts = Counter()
    timeline       = []

    # For each clip: gen npy + per‐hit predict 
    for clip in sorted(os.listdir(RALLY_OUTPUT_DIR)):
        clip_dir = RALLY_OUTPUT_DIR / clip
        print(f"\n[TemPose] Processing {clip} …")
        # ① generate all the .npy required for TemPose
        process_clip(clip, str(clip_dir), T_max=mcfg["sequence_length"])

        # ② do per‐hit classification
        hit_csv = clip_dir / f"{clip}_hits.csv"
        if not hit_csv.exists():
            print(f"[WARN] {hit_csv} not found; skipping.")
            continue

        # This returns a list of (frame_idx, stroke_name) for all hits in this clip
        events = per_hit_predict(model, clip_dir, hit_csv, cfg, device)

        # accumulate
        video_file = clip_dir / f"{clip}.mp4"
        team_file = Path(f"{clip}_teams.csv")
        team_path = clip_dir / team_file

        # Read team data and determine Top/Bottom mapping
        if team_path.exists():
            df_team = pd.read_csv(team_path)

            # Assume y1 < y2 => player is on top (since y grows downward)
            first_frame = df_team[df_team['frame'] == df_team['frame'].min()]
            if first_frame.empty:
                top_player = 0
                bottom_player = 1
            else:
                top_player = first_frame.loc[first_frame['y1'].idxmin(), 'player_id']
                bottom_player = first_frame.loc[first_frame['y1'].idxmax(), 'player_id']
        else:
            print(f"[WARN] {team_path} not found; defaulting player0=Top, player1=Bottom")
            top_player = 0
            bottom_player = 1

        for frame_idx, stroke in events:
            # Map Top_ / Bottom_ to Player0_ / Player1_
            if stroke.startswith("Top_"):
                true_player = top_player
                stroke = stroke.replace("Top_", f"Player{top_player}_")
            elif stroke.startswith("Bottom_"):
                true_player = bottom_player
                stroke = stroke.replace("Bottom_", f"Player{bottom_player}_")
            else:
                true_player = None  # Unknown role

            overall_counts[stroke] += 1
            ts = frame_to_timestamp(video_file, frame_idx, original_video_path=video_path)
            timeline.append({
                "clip": clip,
                "frame": frame_idx,
                "timestamp": ts,
                "stroke": stroke,
                "player_id": true_player
            })
    print("[Message] TemPose finished\n")
    recording_execution_time(logs, "End TemPose")

    # ——— 7. Summarize & print —————————————————————————————————————
    recording_execution_time(logs, "Start Summarize")
    if not timeline:
        print("\nNo hits detected across all clips.")
        return

    # a) Stroke counts
    df_counts = (
        pd.DataFrame.from_records(
            [{"stroke": s, "count": c} for s, c in overall_counts.items()]
        )
        .sort_values("stroke")
        .reset_index(drop=True)
    )
    print("\n=== Stroke counts ===")
    print(df_counts.to_string(index=False))

    # b) Full hit timeline
    df_tl = pd.DataFrame(timeline)
    df_tl = df_tl.sort_values(["clip", "frame"]).reset_index(drop=True)
    print("\n=== Hit timeline ===")
    print("  timestamp stroke")
    # for entry in timeline:
    #    print(f"{entry['timestamp']}   {entry['stroke']}")

    print("\n[Message] Visualizing hits in video...")
    output_video_dir = RESULT_OUTPUT_DIR
    output_video_path = output_video_dir / f"{name}_annotated.mp4"
    visualize_hits_in_video(video_path, timeline, output_path=output_video_path)
    
    # Optionally save CSVs:
    #df_counts.to_csv(RALLY_OUTPUT / "summary_counts.csv", index=False)
    #df_tl   .to_csv(RALLY_OUTPUT / "hit_timeline.csv",  index=False)

    #print(f"\n[Done] Summaries written to {RALLY_OUTPUT / 'summary_counts.csv'} and hit_timeline.csv")
    recording_execution_time(logs, "End Summarize")
    print_execution_time_with_plot(logs, output_video_dir)
    print("\n[All done]")

if __name__ == "__main__":
    main()
