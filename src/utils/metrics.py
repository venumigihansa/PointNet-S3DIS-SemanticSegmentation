import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_accuracy(pred, target):
    """Calculate point-wise accuracy"""
    pred_choice = pred.data.max(2)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    return correct.item() / (target.size(0) * target.size(1))

def calculate_iou(pred, target, num_classes):
    """Calculate mean IoU"""
    pred = pred.data.max(2)[1].cpu().numpy()
    target = target.cpu().numpy()

    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        if target_cls.sum() == 0:  # No ground truth for this class
            if pred_cls.sum() == 0:  # No prediction either
                ious.append(1.0)
            else:  # False positives
                ious.append(0.0)
        else:
            intersection = (pred_cls & target_cls).sum()
            union = (pred_cls | target_cls).sum()
            ious.append(intersection / union)

    return np.mean(ious)

def evaluate_model(model, test_loader, device, class_names):
    """Detailed evaluation of the model"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            data = data.transpose(2, 1)

            pred, _, _ = model(data)
            pred_choice = pred.data.max(2)[1]

            all_preds.append(pred_choice.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Calculate per-class IoU
    class_ious = []
    for cls in range(len(class_names)):
        pred_cls = (all_preds == cls)
        target_cls = (all_targets == cls)

        if target_cls.sum() == 0:
            if pred_cls.sum() == 0:
                class_ious.append(1.0)
            else:
                class_ious.append(0.0)
        else:
            intersection = (pred_cls & target_cls).sum()
            union = (pred_cls | target_cls).sum()
            class_ious.append(intersection / union)

    # Print results
    print("\nPer-class IoU:")
    print("-" * 40)
    for i, (name, iou) in enumerate(zip(class_names, class_ious)):
        print(f"{name:12s}: {iou:.4f}")
    print("-" * 40)
    print(f"Mean IoU: {np.mean(class_ious):.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names)

    return np.mean(class_ious), class_ious

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()