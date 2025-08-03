import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_epoch(model, train_loader, optimizer, criterion, device, feature_transform=False):
    model.train()
    total_loss = 0
    total_acc = 0
    total_iou = 0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        data = data.transpose(2, 1)  # [B, 3, N]

        optimizer.zero_grad()

        pred, trans_input, trans_feat = model(data)

        # Reshape for loss calculation
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)

        # Main loss
        loss = criterion(pred, target)

        # Regularization loss for transformation matrices
        if feature_transform and trans_feat is not None:
            reg_loss = feature_transform_regularizer(trans_feat)
            loss += 0.001 * reg_loss

        loss.backward()
        optimizer.step()

        # Calculate metrics
        pred_reshaped = pred.view(data.size(0), -1, pred.size(-1))
        target_reshaped = target.view(data.size(0), -1)

        acc = calculate_accuracy(pred_reshaped, target_reshaped)
        iou = calculate_iou(pred_reshaped, target_reshaped, model.num_classes)

        total_loss += loss.item()
        total_acc += acc
        total_iou += iou

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc:.4f}',
            'mIoU': f'{iou:.4f}'
        })

    return total_loss / num_batches, total_acc / num_batches, total_iou / num_batches

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_iou = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            data = data.transpose(2, 1)  # [B, 3, N]

            pred, _, _ = model(data)

            # Reshape for loss calculation
            pred_flat = pred.view(-1, pred.size(-1))
            target_flat = target.view(-1)

            loss = criterion(pred_flat, target_flat)

            # Calculate metrics
            acc = calculate_accuracy(pred, target)
            iou = calculate_iou(pred, target, model.num_classes)

            total_loss += loss.item()
            total_acc += acc
            total_iou += iou

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.4f}',
                'mIoU': f'{iou:.4f}'
            })

    return total_loss / num_batches, total_acc / num_batches, total_iou / num_batches

def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        train_ious, val_ious):
    """Plot training and validation curves"""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc')
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # mIoU
    axes[2].plot(epochs, train_ious, 'b-', label='Train mIoU')
    axes[2].plot(epochs, val_ious, 'r-', label='Val mIoU')
    axes[2].set_title('Training and Validation mIoU')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('mIoU')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()