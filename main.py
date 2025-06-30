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
from TrackNetV3.predict       import predict_traj
from TrackNetV3.denoise       import smooth
from MMPose.detect_pose        import process_pose
from HitNet.predict           import predict as hitnet_detect

from TemPose.generate_npy     import process_clip
from TemPose.predict_by_hit   import per_hit_predict, slice_and_pad, get_stroke_types
from TemPose.TemPoseII        import TemPoseII_TF

from team_classfier.sport_player_team_classifier import predict_teams, train_yolo

def frame_to_timestamp(video_file: Path, frame_idx: int, original_video_path: Path = None):
    """Convert frame index to timestamp, relative to original video if provided"""
    # Get clip info
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    clip_name = video_file.parent.name  # e.g. "clip_3"
    clip_num = int(clip_name.split('_')[1])
    
    # Read the timepoints file to get start time in original video
    timepoints_path = Path('rally_clipping/final_result') / f"{original_video_path.stem}.txt"
    
    if original_video_path and timepoints_path.exists():
        with open(timepoints_path, 'r', encoding='utf-8') as f:
            time_points = [line.strip() for line in f.readlines()]
        
        if clip_num <= len(time_points) - 1:
            start_time_str = time_points[clip_num - 1]
            
            try:
                # Handle HH:MM:SS.msec format (like 00:00:02.383333)
                if start_time_str.count(':') == 2:
                    hours, minutes, sec_parts = start_time_str.split(':')
                    seconds, msec = sec_parts.split('.') if '.' in sec_parts else (sec_parts, '0')
                    
                    start_seconds = (int(hours) * 3600) + (int(minutes) * 60) + int(seconds)
                    if msec:
                        start_seconds += float(f"0.{msec}")
                
                # Try parsing as float
                elif '.' in start_time_str and ':' not in start_time_str:
                    start_time_float = float(start_time_str)
                    start_mins = int(start_time_float)
                    start_secs = int((start_time_float - start_mins) * 60)
                    start_seconds = start_mins * 60 + start_secs
                
                # Try MM:SS format
                elif ':' in start_time_str:
                    start_mins, start_secs = map(int, start_time_str.split(':'))
                    start_seconds = start_mins * 60 + start_secs
                
                else:
                    # Handle unexpected format
                    print(f"Warning: Could not parse timestamp {start_time_str}")
                    start_seconds = 0
                    
            except ValueError as e:
                print(f"Warning: Error parsing timestamp {start_time_str}: {e}")
                start_seconds = 0
            
            # Add clip's internal time to start time
            seconds = start_seconds + (frame_idx / fps)
            
            # Format with hours, minutes, seconds and milliseconds
            h, remainder = divmod(seconds, 3600)
            m, s = divmod(remainder, 60)
            int_s = int(s)
            ms = (s - int_s) * 1000000
            
            # Return in format HH:MM:SS.msec
            return f"{int(h):02d}:{int(m):02d}:{int_s:02d}.{int(ms):06d}"
    
    # Fallback to clip-relative timestamp in the same format
    seconds = frame_idx / fps
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    int_s = int(s)
    ms = (s - int_s) * 1000000
    
    return f"{int(h):02d}:{int(m):02d}:{int_s:02d}.{int(ms):06d}"

def visualize_hits_in_video(video_path, timeline, output_path=None):
    """
    Annotate the original video with hit timestamps and stroke types.
    
    Args:
        video_path: Path to the original video
        timeline: List of dictionaries containing hit information
        output_path: Path to save the annotated video (defaults to 'output_annotated.mp4')
    """
    if output_path is None:
        output_path = Path('output_annotated.mp4')
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Convert timestamps to frame numbers
    hit_frames = {}
    for hit in timeline:
        timestamp = hit['timestamp']
        stroke = hit['stroke']
        
        # Parse timestamp (HH:MM:SS.msec)
        hours, minutes, sec_parts = timestamp.split(':')
        seconds, msec = sec_parts.split('.')
        
        total_seconds = (int(hours) * 3600) + (int(minutes) * 60) + int(seconds) + float(f"0.{msec}")
        frame_num = int(total_seconds * fps)
        
        hit_frames[frame_num] = stroke
    
    # Process each frame
    print(f"\n[Message] Annotating video with hit events")
    progress_interval = max(1, total_frames // 100)  # Update progress every 1%
    
    current_frame = 0
    active_hits = []  # List of (hit_text, remaining_display_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame has a hit
        if current_frame in hit_frames:
            stroke = hit_frames[current_frame]
            timestamp = f"{int(current_frame/fps/60):02d}:{int(current_frame/fps)%60:02d}.{int((current_frame/fps%1)*1000):03d}"
            hit_text = f"{timestamp} - {stroke}"
            active_hits.append((hit_text, int(fps * 3)))  # Display for 3 seconds
        
        # Draw active hits info
        y_offset = 50
        for i, (text, remaining) in enumerate(active_hits):
            cv2.putText(frame, text, (50, y_offset + i*40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update active hits (decrement display time)
        active_hits = [(text, remain-1) for text, remain in active_hits if remain > 1]
        
        # Write the frame to output video
        out.write(frame)
        
        # Update progress
        current_frame += 1
        if current_frame % progress_interval == 0:
            print(f"Progress: {current_frame/total_frames*100:.1f}%", end='\r')
    
    cap.release()
    out.release()
    print(f"\n[Message] Annotated video saved to {output_path}")
    return output_path

def main():
    # ——— 1. Paths & config —————————————————————————————————————
    video_path = Path('match1.mp4')
    name       = video_path.stem
    RALLY_OUTPUT_DIR = Path('videos') / name

    COURT_DET_EXE = 'court_detection/court-detection.exe'
    COURT_OUTPUT  = 'court_detection/court.txt'
    COURT_IMAGE   = 'court_detection/court_image.png'
    
    # ——— 2. Rally clipping ——————————————————————————————————————————————
    print("\n[Message] Start rally clipping\n")
    timepoints_clipping(video_path)
    print("[Message] Rally clipping finished\n")
    
    # ——— 3. Court Detection ——————————————————————————————————————————
    print("\n[Message] Start court detection\n")
    subprocess.run(
        [COURT_DET_EXE, str(video_path), COURT_OUTPUT, COURT_IMAGE, "400"], 
        capture_output=True, text=True
    )
    print("[Message] Court detection finished\n")
    
    # ——— 4. Trajectory & Pose Prediction —————————————————————————————————————
    print("\n[Message] Start trajectory & pose prediction\n")
    inferencer = MMPoseInferencer('human')
    for clip in os.listdir(RALLY_OUTPUT_DIR):
        clip_dir  = RALLY_OUTPUT_DIR / clip
        clip_path = clip_dir / f"{clip}.mp4"

        traj_csv = predict_traj(clip_path, str(clip_dir))
        df       = pd.read_csv(traj_csv, encoding="utf-8")
        # smooth(traj_csv, df)

        process_pose(inferencer, clip_path, str(clip_dir), COURT_OUTPUT)
    print("[Message] Trajectory & pose prediction finished\n")
    
    # ——— 5. HitNet ————————————————————————————————————————
    print("\n[Message] Start hit detection\n")
    hitnet_detect(RALLY_OUTPUT_DIR)
    print("[Message] Hit detection finished\n")
    
    # ——— 6. Team Classification ——————————————————————————————————————
    classifier = train_yolo(video_path)
    for clip in os.listdir(RALLY_OUTPUT_DIR):
        clip_dir = RALLY_OUTPUT_DIR / clip
        print(f"\n[Team] Processing {clip} …")
        predict_teams(clip_dir, clip, classifier)

    # ——— 7. TemPose  ——————————————————————————————————————
    # load model config
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

    # ——— 8. Summarize & print —————————————————————————————————————
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
    for entry in timeline:
        print(f"{entry['timestamp']}   {entry['stroke']}")

    print("\n[Message] Visualizing hits in video...")
    output_video_path = Path('videos') / f"{name}_annotated.mp4"
    visualize_hits_in_video(video_path, timeline, output_path=output_video_path)
    
    # Optionally save CSVs:
    #df_counts.to_csv(RALLY_OUTPUT / "summary_counts.csv", index=False)
    #df_tl   .to_csv(RALLY_OUTPUT / "hit_timeline.csv",  index=False)

    #print(f"\n[Done] Summaries written to {RALLY_OUTPUT / 'summary_counts.csv'} and hit_timeline.csv")
    print("\n[All done]")

if __name__ == "__main__":
    main()
