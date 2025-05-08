from rally_clipping.get_result import timepoints_clipping
from pathlib import Path
import subprocess

COURT_DETECTION_PATH = 'court-detection.exe'
COURT_OUTPUT = 'court_detection/court.txt'
COURT_IMAGE = 'court_detection/court_image.png'

if __name__ == '__main__':

    video_path = Path('sample.mp4')

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
    print("The court information is stored in court_detection/")
    print("[Message] Court detection finished\n")