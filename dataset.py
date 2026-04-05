import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import os
from args import get_args
from utils import resize_box_xyxy
import augmentations as aug

class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)

        if transform is None:
            self.transform = aug.NoTransform
        else:
            self.transform = aug.Compose(transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        args = get_args()
        row = self.df.iloc[idx]

        img = Image.open(row['input']).convert("RGB")
        w, h = img.size
        img = img.resize((args.image_size, args.image_size))
        image = to_tensor(img)

        boxes, labels = [], []
        with open(row["output"]) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                x1, y1, x2, y2 = resize_box_xyxy((x1, y1, x2, y2), w, h, args.image_size, args.image_size)
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        
        image, target = self.transform(image, target)
        
        return image, target