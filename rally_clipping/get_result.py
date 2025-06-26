from rally_clipping.module import *
from rally_clipping.img_predict import run_inference
from tqdm import tqdm
import numpy as np
import cv2

def determine_cut_points(
    a: np.ndarray,
    th=0.5,
    steep_th=0.25
):
    points = []

    sp_state = True
    pick_triggered = False

    for i in range(3, len(a)-3):
        if i == 3 and a[i] >= th:
            points.append(i)
            sp_state = False
            continue

        if sp_state:
            if (np.mean(a[i:i+3]) - np.mean(a[i-3:i])) >= steep_th:
                pick_triggered = True

            if pick_triggered and a[i] >= th:
                points.append(i)
                sp_state = False
                pick_triggered = False

        else:
            if (np.mean(a[i-2:i+1]) - np.mean(a[i+1:i+4])) >= steep_th:
                pick_triggered = True

            if pick_triggered and a[i+1] <= th:
                points.append(i)
                sp_state = True
                pick_triggered = False

    if len(points) % 2 != 0:
        points.append(len(a))

    return np.array(points)

# According to the cut points, we can extract the clips and save them as separate files
def video_clipping(video_path, name, fps=30):
    '''
    video_path: str, path to the video file, i.e., the .mp4 video in mmpose/data
    name: str, name of the video file, it should be same as the name in img_predict.py
    fps: int, frame rate of the video, default is 30
    '''

    # Convert time to frame number
    def time_to_frame(time_str):
        hours, minutes, seconds = map(float, time_str.split(':'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return int(total_seconds * fps)

    # Save clips in videos/{name}
    output_dir = Path(f'videos/{name}')
    timepoints_path = Path(f'rally_clipping/final_result/{name}.txt')
    print(f"[Rally Clipping] Timepoints reading from: {timepoints_path}")

    if not timepoints_path.is_file():
        assert(f"File {timepoints_path} not found!")

    with open(timepoints_path, 'r', encoding='utf-8') as f:
        time_points = [line.strip() for line in f.readlines() if line.strip()]

    if not time_points:
        assert("No valid time points found!")

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Rally Clipping] Starts clipping video")

    clips = []
    pbar = tqdm(total=len(time_points)-1, desc = 'Clipping Rallies')
    for i in range(0, len(time_points)-1, 2):
        start_time = time_points[i].strip()
        start_frame = time_to_frame(start_time)
        end_time = time_points[i + 1].strip()
        end_frame = time_to_frame(end_time)

        if end_frame > total_frames:
            end_frame = total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  

        clip_dir = output_dir / f'clip_{i+1}'

        if not clip_dir.is_dir():
            clip_dir.mkdir(parents=True, exist_ok=True)

        clip_filename = clip_dir / f'clip_{i+1}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(str(clip_filename), fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()

        clips.append(str(clip_filename))
        pbar.update(1)

    cap.release()
    pbar.close()

    print(f'[Rally Clipping] Clips saved to {output_dir}')


def timepoints_clipping(video_path: Path, fps = 30):
    # name = '1 green_men - YONEX French Open 2024 Kunlavut Vitidsarn (THA) [8] vs. Shi Yu Qi (CHN) [2] F'
    # name = '2 green_women - YONEX French Open 2024 An Se Young (KOR) [1] vs Akane Yamaguchi (JPN) [4] F'
    # name = ''
    # name = '10 red_men - HSBC BWF World Tour Finals 2023 Kodai Naraoka (JPN) vs Viktor Axelsen (DEN) Group A'
    
    name = video_path.stem

    print(f'Video name: {name}')
    
    prediction_dir = Path('rally_clipping/final_result')
    if not prediction_dir.is_dir():
        prediction_dir.mkdir()
    save_path = prediction_dir/(str(name)+'.txt')
    
    # Skip if save_path already exists
    if save_path.exists():
        print(f"File {save_path} already exists. Skipping processing.")
        return
    
    run_inference(video_path)

    npy_dir = Path('rally_clipping/npy')
    npy_file = npy_dir/(str(name)+'.npy')

    print(f'File name: {npy_file}')

    model_preds = np.load(str(npy_file))

    print(f'Model Prediction: {model_preds}')

    points = determine_cut_points(
        model_preds,
        steep_th=0.25
    )

    # Change to time string
    frameNum_2_time_partial = partial(frameNum_2_time, fps=fps)
    cut_point_time_strs = list(map(frameNum_2_time_partial, points))

    with save_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(cut_point_time_strs))

    videos_dir = Path('videos')
    if not videos_dir.is_dir():
        videos_dir.mkdir()
    
    # video_path = infer_video_path in img_predict.py
    # or modify the path here
    video_clipping(video_path, name)