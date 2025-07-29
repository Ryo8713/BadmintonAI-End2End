from typing import Dict
import cv2
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pandas as pd

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return round(fps)

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

def draw_chinese_text(img, text, position, font_path='font.ttf', font_size=24, color=(255, 255, 153)):
    # 轉成 PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    draw.text(position, text, font=font, fill=color)
    
    # 轉回 OpenCV 格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def visualize_hits_in_video(video_path, df_events, output_path=None):
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
    
    # Process each frame
    print(f"\n[Message] Annotating video with hit events")
    progress_interval = max(1, total_frames // 100)  # Update progress every 1%
    
    current_frame = 0
    event_dict = {int(row['start_frame']): {'end_frame': int(row['end_frame']), 'stroke': row.get('stroke')} for _, row in df_events.iterrows()}
    cur_start, cur_end = -1, -1
    cur_stroke = ''

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this frame is the start frame
        if current_frame in event_dict:
            cur_start, cur_end = current_frame, event_dict[current_frame]['end_frame']
            cur_stroke = event_dict[current_frame]['stroke']
        
        # Check if this frame has a hit
        if current_frame >= cur_start and current_frame <= cur_end:

            if cur_stroke == '未知球種':
                cv2.putText(frame, "??", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (0, 0, 255), 4, cv2.LINE_AA)
            else:
                frame = draw_chinese_text(frame, f'{cur_stroke}', position=(50, 50))
        
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

def recording_execution_time(logs: Dict[str, datetime], status: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs[status] = timestamp
    return logs

def print_execution_time_with_plot(logs: Dict[str, datetime], video_dir: Path):
    print(f"\n[Message] Execution time:")
    
    # Step duration statistics
    step_names = []
    step_durations = []
    step_start_times = []
    
    time_format = "%Y-%m-%d %H:%M:%S"
    log_items = list(logs.items())
    for i in range(0, len(log_items) - 1, 2):  # assuming log is in Start/End pairs
        start_status, start_time = log_items[i]
        end_status, end_time = log_items[i + 1]

        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, time_format)
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, time_format)
        
        step_name = start_status.replace("Start ", "")
        duration = (end_time - start_time).total_seconds()
        
        step_names.append(step_name)
        step_durations.append(duration)
        step_start_times.append(start_time)

        print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] {start_status}")
        print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] {end_status}")
        print(f"→ Duration: {duration:.2f} sec\n")

    total_time = sum(step_durations)
    
    # ────── Plot: Bar Chart ──────
    plt.figure(figsize=(10, 5))
    bars = plt.barh(step_names, step_durations, color='skyblue')
    plt.xlabel('Duration (seconds)')
    plt.title('Execution Time per Step')
    plt.grid(axis='x')

    for bar, duration in zip(bars, step_durations):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
        f'{duration:.1f}s', va='center')

    plt.tight_layout()
    plt.savefig(video_dir / "execution_time.png")
    plt.close()


    # ────── Plot: Pie Chart ──────
    plt.figure(figsize=(6, 6))
    plt.pie(
        step_durations,
        labels=step_names,
        autopct='%1.1f%%',  
        startangle=140
    )
    plt.title('Execution Time Proportion')
    plt.tight_layout()
    plt.savefig(video_dir / "execution_time_proportion.png")
    plt.close()

def merge_consecutuve_frame(df_events):
    '''
    If the hit frames are consecutive, only count 1 hit event.
    Remove the "unknown" stroke type then vote for the most frequent stroke type.
    '''
    start_frame, end_frame, stroke, player = [], [], [], []
    cur_start_frame, prev_frame = -1, -1
    cur_stroke, cur_player = [], []

    for i, row in df_events.iterrows():
        frame_idx = int(row['frame'])
        if cur_start_frame == -1:
            cur_start_frame = frame_idx
            if row['stroke'] != '未知球種':
                cur_stroke.append(row['stroke'])
                cur_player.append(row['player'])
            prev_frame = frame_idx

        elif frame_idx == prev_frame + 1:
            if row['stroke'] != '未知球種':
                cur_stroke.append(row['stroke'])
                cur_player.append(row['player'])
            prev_frame = frame_idx

        else:
            start_frame.append(cur_start_frame)
            end_frame.append(prev_frame)
            
            if not cur_stroke:
                stroke.append('未知球種')
                player.append('X')
            else:
                stroke.append(max(set(cur_stroke), key=cur_stroke.count))
                player.append(max(set(cur_player), key=cur_player.count))
            
            cur_start_frame = frame_idx
            cur_stroke = [row['stroke']] if row['stroke'] != '未知球種' else []
            cur_player = [row['player']] if row['stroke'] != '未知球種' else []
            prev_frame = frame_idx

    df_merged_events = pd.DataFrame({'start_frame': start_frame, 'end_frame': end_frame,'stroke': stroke, 'player': player})
    df_stroke_count = df_merged_events.groupby('stroke')['start_frame'].count().reset_index(name='count')

    return df_merged_events, df_stroke_count