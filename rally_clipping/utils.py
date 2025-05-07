import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss

import numpy as np
import bisect
from pathlib import Path
import random
import cv2
from functools import partial
from concurrent.futures import ThreadPoolExecutor, Future

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

from typing import Union


def time_2_frameNum(time_str, fps: Union[int, float]):
    """
    Converts time data (string or numeric components) to frame number based on FPS.

    Parameters
    ----------
    `time_str`
        Can be a string in 'H:M:S.s' or 'M:S.s' format, or numeric values for (hours, minutes, seconds).
    `fps`
        The frames per second of the video.

    Return
    ------
    The frame number (started from 0).
    """

    if isinstance(time_str, str):
        if time_str.count(':') == 1:
            time_str = '0:' + time_str
        try:
            hour, minute, second = map(float, time_str.split(':'))
        except ValueError:
            raise Exception('Invalid time string format')
    else:
        hour, minute, second = time_str

    total_seconds = hour * 3600 + minute * 60 + second
    return int(fps * total_seconds)


def frameNum_2_time(frame_number: int, fps: Union[int, float]) -> str:
    """
    Converts a frame number to time string (HH:MM:SS.ssssss) based on FPS.

    Parameters
    ----------
    `frame_number`
        The frame number (started from 0).
    `fps`
        The frames per second of the video.

    Returns
    -------
    The time string corresponding to the frame number.
    """
    total_seconds = frame_number / fps

    hours = int(total_seconds // 3600)
    minutes = int(total_seconds % 3600 // 60)
    seconds = total_seconds % 60 + 0.5 / fps

    return f"{hours:02d}:{minutes:02d}:{seconds:09.6f}"


def is_time_timeStr_convert_correct(test_frame_len: int, fps: Union[int, float]):
    '''
    Check whether converting frame numbers to time strings and then converting them back are correct or not.
    '''
    for i in range(test_frame_len):
        t_str = frameNum_2_time(i, fps)
        if i != time_2_frameNum(t_str, fps):
            print(f'Errors start at frame {i}.')
            return False
    return True


def read_rally_points_frameNum(file: str, fps: Union[int, float]) -> np.ndarray:
    '''Read the rally points file and convert them into frame numbers.'''
    with open(file, 'r', encoding='utf-8') as sp_file:
        file_str = sp_file.read()
        start_points = np.array(
            [time_2_frameNum(s, fps=fps)
             for s in file_str.split()], dtype=np.int32
        )
    return start_points


def check_rally_points_r_valid(start_points, end_points):
    '''Check the rally data valid or not.'''
    bigger_arr = start_points[1:] > end_points[:-1]
    all_bigger = not np.any(~bigger_arr)
    assert all_bigger, f'The rally data of the video is incorrect!'


def get_label(frame_num, start_points, end_points):
    '''
    Use in generating labeled images.
    Get the label (rally: 1, not rally: 0) from the start points and end points info.
    '''
    s_pos = bisect.bisect_right(start_points, frame_num)
    e_pos = bisect.bisect_left(end_points, frame_num)
    label = int(s_pos!=e_pos)
    return label


def get_images(folder: Path, n, seed=None):
    '''
    Use in RallyImgDataset.
    Get images (court view or non court view) from the image folder.
    '''
    img_paths = sorted(list(folder.glob('*.jpg')))
    
    fixed = random.Random(seed)
    picked_paths = fixed.sample(img_paths, n)
    
    imgs = []
    
    with ThreadPoolExecutor() as executor:
        tasks: list[Future] = []
        for p in picked_paths:
            tasks.append(executor.submit(
                cv2.imread,
                filename=str(p)
            ))
        for task in tasks:
            imgs.append(task.result())
    
    return imgs


@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader
):
    model.eval()
    all_logits = []
    all_labels = []
    for batch, labels in test_loader:
        batch: Tensor = batch.cuda()
        logits: Tensor = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits).squeeze(-1)
    all_labels = torch.cat(all_labels)
    return all_logits, all_labels


def show_ROC_curve(y_true, y_pred, md_serial_no):
    display = RocCurveDisplay.from_predictions(
        y_true,
        y_pred,
        color="darkorange",
        name=f'md{md_serial_no}',
        plot_chance_level=True,
    )
    display.ax_.set(
        xlabel="FPR",
        ylabel="TRR",
        title="ROC curve",
    )
    plt.show()


def FocalLossWithLogits(reduction='mean'):
    return partial(sigmoid_focal_loss, alpha=-1, gamma=2, reduction=reduction)


def scaled_dec(gamma=2):
    def wrapper(func):
        def inner(*arg, **kwarg):
            loss: Tensor = func(*arg, **kwarg)
            return loss * 2**gamma
        return inner
    return wrapper


def FocalLossScaledWithLogits(reduction='mean', gamma=2):
    dec = scaled_dec(gamma=gamma)
    return dec(partial(sigmoid_focal_loss, alpha=-1, gamma=gamma, reduction=reduction))


if __name__ == '__main__':
    pass
