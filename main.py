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
from TemPose.predict_by_hit   import per_hit_predict
from TemPose.TemPoseII        import TemPoseII_TF

from team_classifier.sport_player_team_classifier import predict_teams, train_yolo

from utils import *

def main():
    # ——— 0. Paths & config —————————————————————————————————————
    video_path = Path('full_1.mp4')
    name       = video_path.stem
    RALLY_OUTPUT_DIR = Path('videos') / name
    RESULT_OUTPUT_DIR = Path('results') / name

    COURT_DET_EXE = 'court_detection/court-detection.exe'
    COURT_OUTPUT  = 'court_detection/court.txt'
    COURT_IMAGE   = 'court_detection/court_image.png'

    # take the correct fps
    fps = get_video_fps(str(video_path))

    draw = False # True if you want to visualize the predictions
    logs = {}    # for recording execution time

    # mkdir RESULT_OUTPUT_DIR
    os.makedirs(RESULT_OUTPUT_DIR, exist_ok = True)
    
    # ——— 1. Rally clipping ——————————————————————————————————————————————
    print("\n[Message] Start rally clipping\n")
    recording_execution_time(logs, "Start Rally Clipping")
    start_frame = timepoints_clipping(video_path, fps)
    print("[Message] Rally clipping finished\n")
    recording_execution_time(logs, "End Rally Clipping")
    
    # ——— 2. Court Detection ——————————————————————————————————————————
    print("\n[Message] Start court detection\n")  
    recording_execution_time(logs, "Start Court Detection")

    # check if the txt file to overwrite exists
    # create file if it doesn't exist
    if not os.path.exists(COURT_OUTPUT):
        with open(COURT_OUTPUT, 'w'):
            pass
    
    # take the 100th frame of clip_7 as the input image for court detection
    ## this video is also used by the team classification
    clip_path = RALLY_OUTPUT_DIR / "clip_7" / f"clip_7.mp4"
    subprocess.run(
        [COURT_DET_EXE, str(clip_path), COURT_OUTPUT, COURT_IMAGE, "100"], 
        capture_output=True, text=True
    )
    print("[Message] Court detection finished\n")
    recording_execution_time(logs, "End Court Detection")
    
    # ——— 3. Trajectory & Pose Prediction —————————————————————————————————————
    print("\n[Message] Start trajectory & pose prediction\n")
    recording_execution_time(logs, "Start Trajectory & Pose Prediction")
    track_model = load_tracknet_model()
    inferencer = MMPoseInferencer('human', device='cuda')
    for clip in os.listdir(RALLY_OUTPUT_DIR):
        clip_dir  = RALLY_OUTPUT_DIR / clip
        clip_path = clip_dir / f"{clip}.mp4"
        start_frame_number = start_frame[clip]

        traj_csv = predict_traj(clip_path, str(clip_dir), track_model, start_frame_number=start_frame_number, verbose=False, draw=draw)

        ## Consideration: unable smooth is acceptable
        # df       = pd.read_csv(traj_csv, encoding="utf-8")
        # smooth(traj_csv, df)

        process_pose(inferencer, clip_path, str(clip_dir), COURT_OUTPUT, start_frame_number, draw)
    print("[Message] Trajectory & pose prediction finished\n")
    recording_execution_time(logs, "End Trajectory & Pose Prediction")
    
    # ——— 4. HitNet ————————————————————————————————————————
    print("\n[Message] Start hit detection\n")
    recording_execution_time(logs, "Start Hit Detection")
    hitnet_detect(RALLY_OUTPUT_DIR, start_frame)
    print("[Message] Hit detection finished\n")
    recording_execution_time(logs, "End Hit Detection")
    
    # ——— 5. Team Classification ——————————————————————————————————————
    ## Consideration: this video is also used by court detection
    print("\n[Message] Start team classification\n")
    recording_execution_time(logs, "Start Team Classification")
    clip_path = RALLY_OUTPUT_DIR / "clip_7" / f"clip_7.mp4"
    classifier = train_yolo(clip_path)
    for clip in os.listdir(RALLY_OUTPUT_DIR):
        clip_dir = RALLY_OUTPUT_DIR / clip
        print(f"\n[Team] Processing {clip} …")
        start_frame_number = start_frame[clip]
        predict_teams(clip_dir, clip, classifier, start_frame_number, draw)
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
    timeline       = []
    all_hit_frames = []
    all_strokes    = []
    all_players    = []

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
        hit_frames, strokes = per_hit_predict(model, clip_dir, hit_csv, cfg, device)

        # accumulate
        team_file = Path(f"{clip}_teams.csv")
        team_path = clip_dir / team_file
        top_bbox_file = clip_dir / f"{clip}_top_bbox.csv"
        bottom_bbox_file = clip_dir / f"{clip}_bottom_bbox.csv"

        # Read team data and determine Top/Bottom mapping
        if team_path.exists():
            df_team = pd.read_csv(team_path)
            df_top_bbox = pd.read_csv(top_bbox_file)
            df_bottom_bbox = pd.read_csv(bottom_bbox_file)

        else:
            print(f"[WARN] {team_path} not found")

        for hit_frame, stroke in zip(hit_frames, strokes):
            # Map Top_ / Bottom_ to Player0_ / Player1_
            true_player = 'X'
            if stroke != "未知球種":
                bbox_stroke = None
                if stroke.startswith("Top_"):
                    x1, y1, x2, y2 = df_top_bbox.loc[df_top_bbox['frame'] == hit_frame, ['x1', 'y1', 'x2', 'y2']].values[0]
                    bbox_stroke = (x1, y1, x2, y2)
                elif stroke.startswith("Bottom_"):
                    x1, y1, x2, y2 = df_bottom_bbox.loc[df_bottom_bbox['frame'] == hit_frame, ['x1', 'y1', 'x2', 'y2']].values[0]
                    bbox_stroke = (x1, y1, x2, y2)

                # Determine true player by checking bbox overlap
                team_frame = df_team[df_team['frame'] == hit_frame]
                best_iou = 0
                for i, row in team_frame.iterrows():
                    team_id = row['team_id']
                    x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']]
                    bbox_player = (x1, y1, x2, y2)
                    iou = calculate_iou(bbox_stroke, bbox_player)
                    if iou > best_iou:
                        true_player = team_id
                        best_iou = iou

            all_players.append(true_player)

            # Not use now
            '''
            ts = frame_to_timestamp(video_file, frame_idx, original_video_path=video_path)
            timeline.append({
                "clip": clip,
                "frame": frame_idx,
                "timestamp": ts,
                "stroke": stroke,
                "player_id": true_player
            })
            '''

        all_hit_frames.extend(hit_frames)
        all_strokes.extend(strokes)

    print("[Message] TemPose finished\n")
    recording_execution_time(logs, "End TemPose")
    
    # ——— 7. Summarize & print —————————————————————————————————————
    recording_execution_time(logs, "Start Summarize")
    '''
    if not timeline:
        print("\nNo hits detected across all clips.")
        return
    '''
    # hit_frames & strokes & players
    df_events = pd.DataFrame({"frame": all_hit_frames, "stroke": all_strokes, "player": all_players}).sort_values("frame").reset_index(drop=True)
    df_merged_events, df_stroke_count = merge_consecutuve_frame(df_events)
    
    # a) Stroke counts
    print("\n=== Stroke counts ===")
    for row in df_stroke_count.itertuples(index=False):
        print(f"{row.stroke:<20} {row.count:>5}")

    # b) Full hit timeline
    output_video_dir = RESULT_OUTPUT_DIR
    
    # Optionally save CSVs:
    df_stroke_count.to_csv(RESULT_OUTPUT_DIR / "summary_counts.csv", index=False)
    df_merged_events.to_csv(RESULT_OUTPUT_DIR / "hit_events.csv",  index=False)

    # Annotate video with hit events
    '''
    df_tl = pd.DataFrame(timeline)
    df_tl = df_tl.sort_values(["clip", "frame"]).reset_index(drop=True)
    print("\n=== Hit timeline ===")
    print("  timestamp stroke")
    # for entry in timeline:
    #    print(f"{entry['timestamp']}   {entry['stroke']}")
    '''
    print("\n[Message] Visualizing hits in video...")
    
    output_video_path = output_video_dir / f"{name}_annotated.mp4"
    visualize_hits_in_video(video_path, df_merged_events, output_path=output_video_path)

    #print(f"\n[Done] Summaries written to {RALLY_OUTPUT / 'summary_counts.csv'} and hit_timeline.csv")
    recording_execution_time(logs, "End Summarize")
    print_execution_time_with_plot(logs, output_video_dir)
    print("\n[All done]")

if __name__ == "__main__":
    main()
