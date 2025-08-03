import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3d(nn.Module):
    """3D Spatial Transformer Network for input point cloud transformation"""
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Add identity matrix
        identity = torch.eye(3, dtype=torch.float32, device=x.device)
        identity = identity.view(1, 9).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, 3, 3)

        return x

class STNkd(nn.Module):
    """k-dimensional Spatial Transformer Network"""
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Add identity matrix
        identity = torch.eye(self.k, dtype=torch.float32, device=x.device)
        identity = identity.view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)

        return x

class PointNetFeatureExtractor(nn.Module):
    """PointNet feature extraction backbone"""
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetFeatureExtractor, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        # Input transformation
        self.input_transform = STN3d()

        # First set of convolutions
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Feature transformation
        if self.feature_transform:
            self.feature_transform_net = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]

        # Input transformation
        trans_input = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_input)
        x = x.transpose(2, 1)

        # First convolution
        x = F.relu(self.bn1(self.conv1(x)))

        # Feature transformation
        if self.feature_transform:
            trans_feat = self.feature_transform_net(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # Store point features for segmentation
        point_feat = x

        # Continue feature extraction
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans_input, trans_feat
        else:
            # For segmentation, concatenate global and local features
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, point_feat], 1), trans_input, trans_feat

class PointNetSegmentation(nn.Module):
    """PointNet for semantic segmentation"""
    def __init__(self, num_classes, feature_transform=False):
        super(PointNetSegmentation, self).__init__()
        self.num_classes = num_classes
        self.feature_transform = feature_transform

        # Feature extractor
        self.feat = PointNetFeatureExtractor(global_feat=False,
                                           feature_transform=feature_transform)

        # Segmentation head
        self.conv1 = nn.Conv1d(1088, 512, 1)  # 1024 + 64 = 1088
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size = x.size()[0]
        n_pts = x.size()[2]

        # Extract features
        x, trans_input, trans_feat = self.feat(x)

        # Segmentation layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.num_classes), dim=-1)
        x = x.view(batch_size, n_pts, self.num_classes)

        return x, trans_input, trans_feat

def feature_transform_regularizer(trans):
    """Regularization loss for feature transformation matrix"""
    d = trans.size()[1]
    batch_size = trans.size()[0]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss