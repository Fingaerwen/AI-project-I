import os
import torch
import torch.optim as optim
import matplotlib
# Use a non-GUI backend only when no display is available.
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from args import get_args


def compute_detection_accuracy(preds, targets, iou_threshold=0.5):
    total_gt = 0
    total_matched = 0

    for pred, target in zip(preds, targets):
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        keep = pred_scores >= 0.5
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        total_gt += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue
        if len(pred_boxes) == 0:
            continue

        iou = box_iou(gt_boxes, pred_boxes)

        matched_pred = set()
        for gt_idx in range(len(gt_boxes)):
            best_iou, best_pred_idx = iou[gt_idx].max(0)
            best_pred_idx = best_pred_idx.item()
            if (best_iou.item() >= iou_threshold
                    and best_pred_idx not in matched_pred
                    and pred_labels[best_pred_idx] == gt_labels[gt_idx]):
                total_matched += 1
                matched_pred.add(best_pred_idx)

    if total_gt == 0:
        return 0.0
    return total_matched / total_gt


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    inter_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    return inter_area / union_area.clamp(min=1e-6)


def save_plots(train_losses, val_losses, val_accuracies, out_dir):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Summary', fontsize=14, fontweight='bold')

    ax1.plot(epochs, train_losses, 'b-o', markersize=4, label='Train Loss')
    ax1.plot(epochs, val_losses,   'r-o', markersize=4, label='Val Loss')
    ax1.set_title('Train vs Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(epochs, [a * 100 for a in val_accuracies], 'g-o', markersize=4, label='Val Accuracy')
    ax2.set_title('Validation Detection Accuracy\n(IoU ≥ 0.5, score ≥ 0.5)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def show_batch(images, targets, max_images=2, block=False, display_seconds=1.5):
    num_images = min(len(images), max_images)
    for i in range(num_images):
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        boxes = targets[i]["boxes"].detach().cpu().numpy()
        labels = targets[i]["labels"].detach().cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                f"class {label}",
                fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )

        ax.set_title(f"Sample {i + 1} in batch")
        ax.axis("off")
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(display_seconds)
            plt.close(fig)


def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_val_loss = float('inf')
    train_losses, val_losses, val_accuracies = [], [], []
    batch_preview_shown = False

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

            if not batch_preview_shown:
                show_batch(images, targets, max_images=2, block=False, display_seconds=1.5)
                batch_preview_shown = True

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            loss.backward()
            optimizer.step()

            batch_size = len(images)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_epoch_loss = running_loss / total_samples

        val_epoch_loss, val_acc = validate_model(model, val_loader, device)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} | "
              f"Val Accuracy: {val_acc * 100:.2f}%")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)


    save_plots(train_losses, val_losses, val_accuracies, args.out_dir)


def validate_model(model, val_loader, device):
    model.train()
    val_loss_sum = 0.0
    total_samples = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device, dtype=torch.float32) for img in images]
            targets_device = [
                {
                    'boxes': t['boxes'].to(device, dtype=torch.float32),
                    'labels': t['labels'].to(device, dtype=torch.int64),
                }
                for t in targets
            ]

            loss_dict = model(images, targets_device)
            loss = sum(loss_value for loss_value in loss_dict.values())

            batch_size = len(images)
            val_loss_sum += loss.item() * batch_size
            total_samples += batch_size

            model.eval()
            preds = model(images)
            model.train()

            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets_device]
            all_preds.extend(preds_cpu)
            all_targets.extend(targets_cpu)

    if total_samples == 0:
        return float('inf'), 0.0

    val_loss = val_loss_sum / total_samples
    val_acc = compute_detection_accuracy(all_preds, all_targets)
    return val_loss, val_acc