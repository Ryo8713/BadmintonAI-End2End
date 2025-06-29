# 羽毛球擊球類型辨識端到端系統

## Installation

It's better to use virtual environment to install the required packages and run the program.
```bash
python -m venv venv
venv/Scripts/activate    # for Windows
pip install requirements -r requirements.txt
```

### MMCV

We recommend to use `openmim` (which is already included in `requirements.txt`) to install `mmcv`. Run the following command:
```bash
mim install mmcv==2.1.0
```

## Usage

Modify `sample.mp4` to your own video path in `main.py`:
```py
video_path = Path('sample.mp4')
```
Currently, the program can only process one video at one time. The result will be store in the `videos/` folder. Take `sample.mp4` as example, the organization of folder may look like:
```
/videos
│── /sample               
│   │── /clip_1
│   │   │── clip_1_ball.csv                  # trajectory
│   │   │── clip_1_bottom.csv                # pose of bottom player
│   │   │── clip_1_pred.mp4                  # visualization of pose
│   │   │── clip_1_top.csv                   # pose of top player
│   │   │── clip_1_visualized_output.csv     # visualization of trajectory
│   │   │── clip_1.mp4                       # original video
│   │   │── clip_1_hits.mp4                  # hit frame (binary)
|   |
|   │── /clip_3 ...

```

## Note

- 上次討論的那個過場畫面的問題還沒改。

- 最後的 TemPose 還沒接上來，如果需要 homography matrix 的話在 `court_detection/court.txt` 裡面可以找到。

- 在 TrackNet & mmpose 那個階段可能會要花一點時間跑，特別是在影片很多的情況下。

- HitNet 現在有 0.86 的 F1-score，應該還行 ?

- 有一個資料夾 `ai_badminton` 是來自 [monotrack](https://github.com/jhwang7628/monotrack) 開發的套件。

