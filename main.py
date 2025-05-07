from rally_clipping.get_result import timepoints_clipping
from pathlib import Path

if __name__ == '__main__':

    video_path = Path('sample.mp4')

    # rally clipping
    print("\n[Message] Start rally clipping\n")
    timepoints_clipping(video_path)
    print("\n[Message] Rally clipping finished\n")