import os
import torch
import torch.optim as optim
from args import get_args

def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, targets in train_loader:
            images = [img.to(device, dtype=torch.float32) for img in images]
            targets = [
                {
                    'boxes': t['boxes'].to(device, dtype=torch.float32),
                    'labels': t['labels'].to(device, dtype=torch.int64),
                }
                for t in targets
            ]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            loss.backward()
            optimizer.step()

            batch_size = len(images)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_epoch_loss = running_loss / total_samples

        val_epoch_loss = validate_model(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)


def validate_model(model, val_loader, device):
    model.eval()
    val_loss_sum = 0.0
    total_samples = 0

    for images, targets in val_loader:
        images = [img.to(device, dtype=torch.float32) for img in images]
        targets = [
            {
                'boxes': t['boxes'].to(device, dtype=torch.float32),
                'labels': t['labels'].to(device, dtype=torch.int64),
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_value for loss_value in loss_dict.values())

        batch_size = len(images)
        val_loss_sum += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return float('inf')

    return val_loss_sum / total_samples