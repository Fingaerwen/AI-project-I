from args import get_args
import pandas as pd
import os
import torch
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
import model
from trainer import train_model


def collect(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    args = get_args()
    print(args)
    
    csv_dir = '.'
    df_train = pd.read_csv(os.path.join(csv_dir, 'Data/train.csv'))
    df_val = pd.read_csv(os.path.join(csv_dir, 'Data/val.csv'))
    
    train_dataset = ObjDetectionDataset(df_train)
    val_dataset = ObjDetectionDataset(df_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collect,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collect,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    built_model = model.buildModel(args.backbone, num_classes=args.num_classes + 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(built_model, train_loader, val_loader, device)

if __name__ == '__main__':
    main()

exit()