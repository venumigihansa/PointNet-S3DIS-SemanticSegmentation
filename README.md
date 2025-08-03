# 🏠 PointNet for S3DIS Scene Semantic Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A complete PyTorch implementation of **PointNet** for 3D indoor scene semantic segmentation using the Stanford 3D Indoor Scene Dataset (S3DIS). This project implements the architecture from scratch based on the original research paper by Qi et al.

## 🎯 Overview

This implementation focuses on **scene semantic segmentation**, classifying every point in room-scale 3D point clouds into semantic categories. The model processes entire indoor scenes and assigns semantic labels to each point, enabling detailed understanding of 3D indoor environments.

## 🏗️ Architecture

### Core Components
- 🔄 **STN3d**: 3D Spatial Transformer Network for input transformation
- ⚙️ **STNkd**: k-dimensional Spatial Transformer Network for feature alignment  
- 🧠 **PointNetFeatureExtractor**: Main feature extraction backbone
- 🎯 **PointNetSegmentation**: Complete segmentation model with classification head

### ✨ Key Features
- 🔁 Input transformation networks for rotation invariance
- 🔧 Optional feature transformation for better alignment
- 📍 Point-wise classification for semantic segmentation
- 📏 Regularization loss for transformation matrices
- 🏷️ Support for 13 semantic classes from S3DIS

## 📊 Dataset

**S3DIS (Stanford 3D Indoor Scene Dataset)**
- 🏢 6 indoor areas with 271 rooms
- 🏷️ 13 semantic classes: `ceiling`, `floor`, `wall`, `beam`, `column`, `window`, `door`, `chair`, `table`, `bookcase`, `sofa`, `board`, `clutter`
- 🌈 Point clouds with RGB information
- 📋 Instance and semantic annotations

## 📁 Project Structure

```
pointnet-s3dis/
├── 📁 src/
│   ├── 🤖 models/
│   │   ├── __init__.py
│   │   ├── pointnet.py          # Core PointNet architecture
│   │   └── transforms.py        # Spatial transformer networks
│   ├── 📊 data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # S3DIS dataset loader
│   │   └── preprocessing.py     # Data preprocessing utilities
│   ├── 🛠️ utils/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── visualization.py     # Visualization utilities
│   │   └── training.py          # Training utilities
│   └── 🚀 train.py              # Main training script
├── 📓 notebooks/
│   └── pointnet_implementation.ipynb
├── ⚙️ configs/
│   └── config.yaml
├── 📋 requirements.txt
├── 📖 README.md
└── 🚫 .gitignore
```

## 🚀 Quick Start

### 1️⃣ Installation
```bash
git clone https://github.com/yourusername/pointnet-s3dis.git
cd pointnet-s3dis
pip install -r requirements.txt
```

### 2️⃣ Data Preparation
```bash
python src/data/preprocessing.py
```

### 3️⃣ Training
```bash
# Default training
python src/train.py

# Custom parameters
python src/train.py --batch_size 16 --num_points 4096 --epochs 100 --test_area 5
```

### 4️⃣ Evaluation
```bash
python src/evaluate.py --model_path checkpoints/best_model.pth --test_area 5
```

### 5️⃣ Visualization
```bash
python src/visualize.py --model_path checkpoints/best_model.pth --num_samples 5
```

## 📈 Results

### 🎯 Performance Metrics

| Metric | Value | Status |
|--------|-------|---------|
| Final Validation Accuracy | **67.45%** | ✅ Good |
| Best Mean IoU | **36.42%** | ✅ Solid |
| Final Mean IoU | **31.41%** | ✅ Reasonable |
| Training Epochs | **100** | ⏱️ Complete |

### 🏆 Per-Class IoU Results

| Class | IoU | Performance | Analysis |
|-------|-----|-------------|----------|
| 🏠 **Floor** | **89.03%** | 🥇 Excellent | Best performing - large planar surfaces |
| 🏠 **Ceiling** | **83.43%** | 🥇 Excellent | Strong geometric consistency |
| 🧱 **Wall** | **54.12%** | 🥈 Good | Solid performance with room for improvement |
| 📚 **Bookcase** | **41.17%** | 🥉 Moderate | Complex furniture structure |
| 🪑 **Table** | **35.24%** | 🥉 Moderate | Shape variation challenges |
| 🪑 **Chair** | **30.61%** | 🥉 Moderate | High variability and occlusion |
| 🚪 **Door** | **26.53%** | 🥉 Moderate | Confusion with walls |
| 🪟 **Window** | **23.24%** | 🥉 Moderate | Embedded in walls |
| 🗑️ **Clutter** | **16.51%** | ⚠️ Poor | Highly variable category |
| 📋 **Board** | **5.97%** | ❌ Very Poor | Small objects, scale issues |
| 🏛️ **Column** | **2.47%** | ❌ Very Poor | Thin structures, limited examples |
| 🏗️ **Beam** | **0.00%** | ❌ Failed | Extremely sparse in dataset |
| 🛋️ **Sofa** | **0.00%** | ❌ Failed | High variation, dataset imbalance |

## 🔧 Implementation Details

### 🏗️ Model Architecture
- **Input**: Point clouds with XYZ coordinates (N × 3)
- **Feature Extraction**: Shared MLPs with batch normalization
- **Spatial Invariance**: Transformer networks for geometric robustness
- **Permutation Invariance**: Global max pooling
- **Output**: Point-wise classification head

### 🎯 Training Strategy
- **Loss**: Cross-entropy with feature transformation regularization
- **Optimizer**: Adam with learning rate scheduling
- **Split**: Area-based (Area 5 for testing)
- **Augmentation**: Point sampling and normalization

### 📊 Hyperparameters
```yaml
🎛️ Training:
  batch_size: 16
  num_points: 4096
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-4

🤖 Model:
  num_classes: 13
  feature_transform: true

📊 Data:
  test_area: 5
```

## 📊 Visualization Tools

The project includes comprehensive visualization capabilities:
- 🌈 RGB point cloud visualization
- 🎨 Semantic segmentation results
- 📊 Confusion matrices
- 📈 Training curve plots
- 📊 Per-class performance analysis

## 📚 References

- 📄 [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- 💻 [Original PointNet Implementation](https://github.com/charlesq34/pointnet)
- 🏠 [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html)

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{qi2017pointnet,
  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1612.00593},
  year={2017}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 👨‍🔬 Original PointNet authors for the groundbreaking architecture
- 🏛️ Stanford University for the S3DIS dataset  
- 🔥 PyTorch team for the deep learning framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:
- 🎯 Data augmentation techniques
- ⚖️ Class balancing strategies
- 🏗️ Architecture enhancements
- 📊 Additional evaluation metrics

## 📞 Contact

For questions or suggestions, please open an issue or contact [your.email@example.com]

---

⭐ **Star this repo if you find it useful!** ⭐
