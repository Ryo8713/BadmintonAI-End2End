from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.models import resnet18
from torchinfo import summary

from rally_clipping.utils import *


class RallyImgDataset(Dataset):
    def __init__(
        self,
        image_folder: Path,
        v_idx_ls: list[int],
        n_imgs_each: int,
        is_train: bool,
        set_seed=True
    ) -> None:
        super().__init__()
        seed = 100 if set_seed else None  # for dataset fixed
        half_n = n_imgs_each // 2

        imgs = []
        for v_idx in v_idx_ls:
            imgs += [
                *get_images(image_folder/f'{v_idx}'/'court_view', half_n, seed),
                *get_images(image_folder/f'{v_idx}'/'non_court_view', half_n, seed)
            ]

        if is_train:
            self.tfm = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.RandomHorizontalFlip()
            ])
        else:
            self.tfm = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.imgs = imgs
        self.labels = torch.cat((
            torch.ones(half_n, dtype=torch.float),
            torch.zeros(half_n, dtype=torch.float)
        )).unsqueeze(0).expand(len(v_idx_ls), -1).flatten()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        img = self.tfm(self.imgs[i])
        return img, self.labels[i]


class CourtViewClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet18 = resnet18(num_classes=1)

    def forward(self, x: Tensor):
        x = self.resnet18(x)
        return x.squeeze(-1)


if __name__ == "__main__":
    # image_folder = Path('image')
    # dataset = RallyImgDataset(
    #     image_folder,
    #     v_idx_ls=[1, 12],
    #     n_imgs_each=1000,
    #     is_train=False,
    #     set_seed=True
    # )
    # print(len(dataset))

    model = CourtViewClassifier()
    summary(model, input_size=(1, 3, 64, 64))

    pass