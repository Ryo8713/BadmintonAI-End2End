# 羽毛球擊球類型辨識端到端系統

我這裡的套件
```
mmpose      1.3.2
numpy       1.26.4
pandas      2.2.3
sckit-learn 1.5.2
torch       2.1.0+cu118
```
## Usage

在 `main.py` 把這行改成影片的路徑，然後執行就可以了。
```py
video_path = Path('sample.mp4')
```
目前只能一次處理一個影片，最後的結果會存在 `videos/`，以 `sample.mp4` 為例，程式跑完之後資料夾可能會長這樣
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
|   │── /clip_2 ...

```

## Note

- 上次討論的那個過場畫面的問題還沒改。

- 最後的 TemPose 還沒接上來，如果需要 homography matrix 的話在 `court_detection/court.txt` 裡面可以找到。

- 在 TrackNet & mmpose 那個階段可能會要花一點時間跑，特別是在影片很多的情況下。

- HitNet 現在有 0.86 的 F1-score，應該還行 ?

- 有一個資料夾 `ai_badminton` 是來自 [monotrack](https://github.com/jhwang7628/monotrack) 開發的套件。

