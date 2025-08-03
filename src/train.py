import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.pointnet import PointNetSegmentation, feature_transform_regularizer
from src.data.dataset import S3DISDataset
from src.utils.training import train_epoch, validate_epoch, plot_training_curves
from src.utils.metrics import evaluate_model, calculate_accuracy, calculate_iou
from src.utils.visualization import visualize_predictions, class_names

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    # Hyperparameters
    batch_size = 16
    num_points = 4096
    num_epochs = 100
    learning_rate = 0.001
    test_area = 5  # Area 5 for testing
    feature_transform = True

    # Data loading
    data_dir = './s3dis_data/processed'
    train_dataset = S3DISDataset(data_dir, num_points=num_points,
                                split='train', test_area=test_area)
    test_dataset = S3DISDataset(data_dir, num_points=num_points,
                               split='test', test_area=test_area)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, drop_last=False)

    # Model, loss, optimizer
    num_classes = 13  # S3DIS has 13 classes
    model = PointNetSegmentation(num_classes=num_classes,
                                feature_transform=feature_transform).to(device)

    # Use weighted loss to handle class imbalance
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")

    # Training loop
    best_iou = 0
    train_losses, train_accs, train_ious = [], [], []
    val_losses, val_accs, val_ious = [], [], []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        # Train
        train_loss, train_acc, train_iou = train_epoch(
            model, train_loader, optimizer, criterion, device, feature_transform
        )

        # Validate
        val_loss, val_acc, val_iou = validate_epoch(
            model, test_loader, criterion, device
        )

        scheduler.step()

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_ious.append(train_iou)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_ious.append(val_iou)

        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, mIoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {val_iou:.4f}')

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_pointnet_s3dis.pth')
            print(f'New best mIoU: {best_iou:.4f} - Model saved!')

    # Save final checkpoint with complete training state
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'hyperparameters': {
            'batch_size': batch_size,
            'num_points': num_points,
            'learning_rate': learning_rate,
            'test_area': test_area,
            'feature_transform': feature_transform,
            'num_classes': num_classes
        }
    }

    torch.save(final_checkpoint, 'final_checkpoint_pointnet_s3dis.pth')
    print(f'\nFinal checkpoint saved with best mIoU: {best_iou:.4f}')

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        train_ious, val_ious)

    return model, (train_losses, val_losses, train_accs, val_accs, train_ious, val_ious)

if __name__ == "__main__":
    print("Starting PointNet training for S3DIS semantic segmentation...")
    print(f"Device: {device}")

    # Train the model
    model, metrics = main()

    # Load best model for evaluation
    model.load_state_dict(torch.load('best_pointnet_s3dis.pth'))

    # Evaluate on test set
    data_dir = './s3dis_data/processed'
    test_dataset = S3DISDataset(data_dir, num_points=4096, split='test', test_area=5)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    mean_iou, class_ious = evaluate_model(model, test_loader, device, class_names)

    print(f"\nTraining completed! Best mIoU: {mean_iou:.4f}")

    print("Visualizing model predictions...")
    print("="*50)

    # Visualize some sample predictions
    visualize_predictions(model, test_dataset, device, num_samples=3)