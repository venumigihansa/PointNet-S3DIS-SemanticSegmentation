import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random
from matplotlib.colors import ListedColormap
import seaborn as sns
from matplotlib.patches import Patch

# S3DIS class names and colors
class_names = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter'
]

# Define distinct colors for each class
class_colors = [
    '#FF6B6B',  # ceiling - red
    '#4ECDC4',  # floor - teal
    '#45B7D1',  # wall - blue
    '#96CEB4',  # beam - green
    '#FFEAA7',  # column - yellow
    '#DDA0DD',  # window - plum
    '#98D8C8',  # door - mint
    '#F7DC6F',  # chair - light yellow
    '#BB8FCE',  # table - light purple
    '#85C1E9',  # bookcase - light blue
    '#F8C471',  # sofa - orange
    '#82E0AA',  # board - light green
    '#D2B4DE'   # clutter - lavender
]

def load_sample_data(data_dir='./s3dis_data/processed'):
    """Load a random sample from the processed S3DIS data"""
    point_files = [f for f in os.listdir(data_dir) if f.endswith('_point.npy')]

    if not point_files:
        print("No processed files found! Make sure to run the preprocessing first.")
        return None, None, None, None

    selected_file = random.choice(point_files)
    base_name = selected_file.replace('_point.npy', '')
    print(f"Loading sample: {base_name}")

    points = np.load(os.path.join(data_dir, f'{base_name}_point.npy'))
    sem_labels = np.load(os.path.join(data_dir, f'{base_name}_sem_label.npy'))
    ins_labels = np.load(os.path.join(data_dir, f'{base_name}_ins_label.npy'))

    return points, sem_labels, ins_labels, base_name

def downsample_points(points, labels, ins_labels, max_points=100000):
    """Downsample points for faster visualization"""
    if len(points) <= max_points:
        return points, labels, ins_labels
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices], labels[indices], ins_labels[indices]

def auto_point_size(num_points):
    """Dynamically determine point size"""
    if num_points > 100000:
        return 0.5
    elif num_points > 50000:
        return 1
    else:
        return 2

def set_equal_aspect(ax, x, y, z, zoom=1.0):
    """Set equal aspect ratio and zoom in for better visualization"""
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0 * zoom
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def visualize_rgb_point_cloud(points, title="RGB Point Cloud"):
    """Visualize point cloud with original RGB colors"""
    s = auto_point_size(len(points))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    colors = points[:, 3:6] / 255.0

    ax.scatter(x, y, z, c=colors, s=s, alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    set_equal_aspect(ax, x, y, z, zoom=0.8)
    plt.tight_layout()
    plt.show()

def visualize_semantic_segmentation(points, sem_labels, title="Semantic Segmentation"):
    """Visualize point cloud colored by semantic classes"""
    s = auto_point_size(len(points))
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(121, projection='3d')

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    colors = [class_colors[int(label)] for label in sem_labels]

    ax1.scatter(x, y, z, c=colors, s=s, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(title)
    set_equal_aspect(ax1, x, y, z, zoom=0.8)

    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    legend_elements = [Patch(facecolor=class_colors[i], label=class_names[i]) for i in range(len(class_names))]
    ax2.legend(handles=legend_elements, loc='center', fontsize=12)
    ax2.set_title('Class Legend', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def visualize_top_down_view(points, sem_labels, title="Top-Down Semantic View"):
    """Visualize top-down view of the room"""
    s = auto_point_size(len(points))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    x, y = points[:, 0], points[:, 1]

    colors_rgb = points[:, 3:6] / 255.0
    ax1.scatter(x, y, c=colors_rgb, s=s, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Top-Down RGB View')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    colors_sem = [class_colors[int(label)] for label in sem_labels]
    ax2.scatter(x, y, c=colors_sem, s=s, alpha=0.7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top-Down Semantic View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_class_distribution(sem_labels, title="Class Distribution"):
    """Plot distribution of semantic classes"""
    unique_labels, counts = np.unique(sem_labels, return_counts=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    class_names_present = [class_names[int(label)] for label in unique_labels]
    colors_present = [class_colors[int(label)] for label in unique_labels]

    bars = ax1.bar(class_names_present, counts, color=colors_present, alpha=0.7)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Points')
    ax1.set_title('Point Count by Class')
    ax1.tick_params(axis='x', rotation=45)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{count:,}', ha='center', va='bottom', fontsize=10)

    ax2.pie(counts, labels=class_names_present, colors=colors_present, autopct='%1.1f%%')
    ax2.set_title('Class Distribution (Percentage)')
    plt.tight_layout()
    plt.show()

def visualize_sample_complete():
    """Complete visualization of a random sample"""
    print("Loading and visualizing S3DIS sample...")
    points, sem_labels, ins_labels, sample_name = load_sample_data()
    if points is None:
        return

    print(f"Sample: {sample_name}")
    print(f"Total points: {len(points):,}")
    print(f"Point cloud bounds:")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print(f"Unique classes: {len(np.unique(sem_labels))}")
    print(f"Unique instances: {len(np.unique(ins_labels))}")
    print("-" * 50)

    points_viz, sem_viz, ins_viz = downsample_points(points, sem_labels, ins_labels, 100000)
    print(f"Downsampled to {len(points_viz):,} points for visualization")

    print("1. RGB Point Cloud Visualization")
    visualize_rgb_point_cloud(points_viz, f"RGB Point Cloud - {sample_name}")

    print("2. Semantic Segmentation Visualization")
    visualize_semantic_segmentation(points_viz, sem_viz, f"Semantic Segmentation - {sample_name}")

    print("3. Top-Down Views")
    visualize_top_down_view(points_viz, sem_viz, f"Top-Down Views - {sample_name}")

    print("4. Class Distribution Analysis")
    plot_class_distribution(sem_labels, f"Class Distribution - {sample_name}")

def visualize_multiple_samples(num_samples=3):
    """Visualize multiple samples for comparison"""
    data_dir = './s3dis_data/processed'
    point_files = [f for f in os.listdir(data_dir) if f.endswith('_point.npy')]
    if len(point_files) < num_samples:
        num_samples = len(point_files)
    selected_files = random.sample(point_files, num_samples)

    fig = plt.figure(figsize=(15, 5 * num_samples))
    for i, file in enumerate(selected_files):
        base_name = file.replace('_point.npy', '')
        points = np.load(os.path.join(data_dir, f'{base_name}_point.npy'))
        sem_labels = np.load(os.path.join(data_dir, f'{base_name}_sem_label.npy'))

        if len(points) > 100000:
            indices = np.random.choice(len(points), 100000, replace=False)
            points = points[indices]
            sem_labels = sem_labels[indices]

        s = auto_point_size(len(points))
        ax1 = fig.add_subplot(num_samples, 2, 2*i + 1, projection='3d')
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        colors = points[:, 3:6] / 255.0
        ax1.scatter(x, y, z, c=colors, s=s, alpha=0.6)
        ax1.set_title(f'{base_name} - RGB')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        set_equal_aspect(ax1, x, y, z, zoom=0.8)

        ax2 = fig.add_subplot(num_samples, 2, 2*i + 2, projection='3d')
        colors_sem = [class_colors[int(label)] for label in sem_labels]
        ax2.scatter(x, y, z, c=colors_sem, s=s, alpha=0.7)
        ax2.set_title(f'{base_name} - Semantic')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        set_equal_aspect(ax2, x, y, z, zoom=0.8)

    plt.tight_layout()
    plt.show()

def visualize_predictions(model, dataset, device, num_samples=3):
    """Visualize model predictions"""
    model.eval()

    fig = plt.figure(figsize=(20, 6 * num_samples))

    for i in range(num_samples):
        # Get random sample
        idx = random.randint(0, len(dataset) - 1)
        data, target = dataset[idx]
        data = data.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            data_input = data.transpose(2, 1)
            pred, _, _ = model(data_input)
            pred_choice = pred.data.max(2)[1]

        # Convert to numpy
        points = data.cpu().numpy()[0]
        target_np = target.cpu().numpy()[0]
        pred_np = pred_choice.cpu().numpy()[0]

        # Plot ground truth
        ax1 = fig.add_subplot(num_samples, 2, 2*i + 1, projection='3d')
        colors_gt = [class_colors[int(label)] for label in target_np]
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors_gt, s=1, alpha=0.7)
        ax1.set_title(f'Sample {i+1} - Ground Truth')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Plot prediction
        ax2 = fig.add_subplot(num_samples, 2, 2*i + 2, projection='3d')
        colors_pred = [class_colors[int(label)] for label in pred_np]
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors_pred, s=1, alpha=0.7)
        ax2.set_title(f'Sample {i+1} - Prediction')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.show()