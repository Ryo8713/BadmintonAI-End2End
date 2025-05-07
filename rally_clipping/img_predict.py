from torch.utils.data import IterableDataset
import torch.nn.functional as F

import moviepy.editor as mpe
from tqdm import tqdm

from rally_clipping.module import *


class RallyImgDataset_infer(IterableDataset):
    def __init__(self, video_path) -> None:
        super().__init__()
        self.video: mpe.VideoClip = mpe.VideoFileClip(video_path).resize((64, 64))
        self.iterator = self.video.iter_frames()

        self.tfm = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            frame = next(self.iterator)
            return self.tfm(frame)
        except StopIteration:
            raise StopIteration()

    def __len__(self):
        return int(self.video.duration * self.video.fps)


@torch.no_grad()
def infer(
    model: CourtViewClassifier,
    loader: DataLoader,
    device: torch.device
):
    model.eval()
    predictions = []
    pbar = tqdm(range(len(loader)), desc='Processing', unit='batch')
    for batch in loader:
        batch: Tensor = batch.to(device)
        pred: Tensor = F.sigmoid(model(batch))
        predictions.append(pred.squeeze(-1).cpu().numpy())
        pbar.update()
    pbar.close()
    return np.concatenate(predictions)


def run_inference(infer_video_path: Path, model_serial_no: int = 3):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    weight_path = Path('rally_clipping/weight')/f'img_md{model_serial_no}.pt'
    # infer_video_path = Path("C:/MyResearch/broadcast_video/5 green_women - YONEX French Open 2024 Tai Tzu Ying (TPE) [3] vs. Aya Ohori (JPN) QF.mp4")

    npy_dir = Path('rally_clipping/npy')
    if not npy_dir.is_dir():
        npy_dir.mkdir()
    save_path = npy_dir/(str(infer_video_path.stem)+'.npy')

    data = RallyImgDataset_infer(str(infer_video_path))
    fps = data.video.fps
    loader = DataLoader(data, batch_size=512, pin_memory=torch.cuda.is_available())

    model = CourtViewClassifier().to(device)
    model.load_state_dict(torch.load(weight_path))

    print('Video:', infer_video_path.name)
    predictions = infer(model, loader, device)

    np.save(str(save_path), predictions)
