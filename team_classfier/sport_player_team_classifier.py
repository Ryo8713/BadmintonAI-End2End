import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from sports.common.team import TeamClassifier

# ====== 設定區 ======
VIDEO_PATH = r"sample.mp4"
OUTPUT_VIDEO_PATH = r"box.mp4"
CSV_PATH = r"box.csv"
COURT_FILE = r"court_detection/court.txt"
FRAME_STRIDE = 30
DEVICE = "cuda"
FRAME_LIMIT = None  # 若要跑完整影片就設為 None
BOX_COLORS = [(0, 0, 255), (255, 0, 0)]  # Team 0: red, Team 1: blue
CONF_THRESH = 0.6  # ← 加入置信度閾值
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 293, 252, 996, 668
# ====================

def set_roi(x1, y1, x2, y2):
    global ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
    ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = x1, y1, x2, y2

def read_corner_set_roi(file_path):

    def parse_point(line_str):
        x_str, y_str = line_str.split(';')
        return float(x_str), float(y_str)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        raise ValueError("Less than 3 lines")
    
    corner1, corner2 = lines[0].strip(), lines[2].strip()
    x1, y1 = parse_point(corner1)
    x2, y2 = parse_point(corner2)
    set_roi(x1, y1, x2, y2)

def box_overlaps_roi(x1, y1, x2, y2):
    return not (x2 < ROI_X1 or x1 > ROI_X2 or y2 < ROI_Y1 or y1 > ROI_Y2)

def get_player_crops(model, video_path, stride=60):
    crops = []
    frame_gen = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    for idx, frame in enumerate(frame_gen):
        result = model(frame, imgsz=640, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        print(f"[Debug] Frame {idx}: Detected {len(detections.xyxy)} objects")
        for xyxy, conf in zip(detections.xyxy, detections.confidence):
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            if box_overlaps_roi(x1, y1, x2, y2):
                crop = sv.crop_image(frame, xyxy)
                crops.append(crop)
    return crops

def draw_boxes(frame, detections, team_ids):
    for xyxy, team_id in zip(detections.xyxy, team_ids):
        x1, y1, x2, y2 = map(int, xyxy)
        if box_overlaps_roi(x1, y1, x2, y2):
            color = BOX_COLORS[int(team_id)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Team {team_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
    return frame

def train_yolo(full_video_path):
    model = YOLO("yolov8s.pt").to(DEVICE)
    classifier = TeamClassifier(device=DEVICE)

    # Consider use the exact corner coordinates of each video
    read_corner_set_roi(COURT_FILE) 

    crops = get_player_crops(model, full_video_path, stride=FRAME_STRIDE)
    print(f"收集到 {len(crops)} 張人物圖像")
    classifier.fit(crops)

    return classifier

def predict_teams(clip_dir, clip, classifier):
    '''
    print("[1] 載入模型...")
    model = YOLO("yolov8s.pt").to(DEVICE)
    classifier = TeamClassifier(device=DEVICE)

    print("[2] 收集 ROI 內人物 crop（重疊保留）...")
    crops = get_player_crops(model, video_path, stride=FRAME_STRIDE)
    print(f"收集到 {len(crops)} 張人物圖像")

    print("[3] 訓練 TeamClassifier...")
    classifier.fit(crops)
    '''
    model = YOLO("yolov8s.pt").to(DEVICE)

    video_path = f'{clip_dir}/{clip}.mp4'
    output_video_path = f'{clip_dir}/{clip}_teams.mp4'
    csv_path = f'{clip_dir}/{clip}_teams.csv'

    print("[4] 開始影片分析與畫圖...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    csv_rows = [["frame", "player_id", "x1", "y1", "x2", "y2", "team_id"]]
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (FRAME_LIMIT is not None and frame_idx > FRAME_LIMIT):
            break

        result = model(frame, imgsz=640, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        valid_xyxy = []
        for xyxy, conf in zip(detections.xyxy, detections.confidence):
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            if box_overlaps_roi(x1, y1, x2, y2):
                valid_xyxy.append(xyxy)

        if valid_xyxy:
            crops = [sv.crop_image(frame, xyxy) for xyxy in valid_xyxy]
            team_ids = classifier.predict(crops)

            for pid, (xyxy, tid) in enumerate(zip(valid_xyxy, team_ids)):
                x1, y1, x2, y2 = map(int, xyxy)
                csv_rows.append([frame_idx, pid, x1, y1, x2, y2, int(tid)])

            detections.xyxy = np.array(valid_xyxy)
            frame = draw_boxes(frame, detections, team_ids)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"[5] 輸出影片：{output_video_path}")
    print(f"[6] 匯出 CSV：{csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
