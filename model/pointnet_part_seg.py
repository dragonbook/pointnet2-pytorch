'''
Modified from https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        iden3 = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1, 9)
        self.register_buffer('iden3', iden3)

    def forward(self, x):
        '''
        x: (batch_size, 3, n)
        '''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (batch_size, 9)

        x = x + self.iden3.repeat(x.shape[0], 1)
        x = x.view(-1, 3, 3)

        return x


class PointNetPartSegmentNet(nn.Module):
    def __init__(self, num_classes, use_stn=True):
        super(PointNetPartSegmentNet, self).__init__()
        self.feat_conv1 = nn.Conv1d(3, 64, 1)
        self.feat_conv2 = nn.Conv1d(64, 128, 1)
        self.feat_conv3 = nn.Conv1d(128, 1024, 1)
        self.feat_bn1 = nn.BatchNorm1d(64)
        self.feat_bn2 = nn.BatchNorm1d(128)
        self.feat_bn3 = nn.BatchNorm1d(1024)

        self.num_classes = num_classes

        self.seg_conv1 = nn.Conv1d(1088, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, self.num_classes, 1)
        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)

        self.use_stn = use_stn
        if self.use_stn:
            self.stn = STN3d()

    def forward(self, x):
        '''
        x: (batch_size, 3, num_points)
        '''
        num_points = x.size()[2]

        if self.use_stn:
            trans = self.stn(x)
            x = x.transpose(1, 2)
            x = torch.bmm(x, trans)
            x = x.transpose(1, 2)

        x = F.relu(self.feat_bn1(self.feat_conv1(x))) 
        point_feature = x  # (batch_size, 64, num_points)
        x = F.relu(self.feat_bn2(self.feat_conv2(x)))
        x = self.feat_bn3(self.feat_conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        global_feature = x.view(-1, 1024)  # (batch_size, 1024)

        global_feature = global_feature.view(-1, 1024, 1).repeat(1, 1, num_points)
        feature = torch.cat([global_feature, point_feature], dim=1)  # (batch_size, 1088, num_points)

        batch_size = feature.size()[0]
        x = F.relu(self.seg_bn1(self.seg_conv1(feature)))
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x = F.relu(self.seg_bn3(self.seg_conv3(x)))
        x = self.seg_conv4(x)  # (batch_size, num_classes, num_points)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.num_classes), dim=-1)
        x = x.view(batch_size, num_points, self.num_classes)

        return x


if __name__ == '__main__':
    input_data = torch.rand(32, 3, 2500)
    print('input_data.shape: {}'.format(input_data.shape))
    seg = PointNetPartSegmentNet(num_classes=10)
    output_data = seg(input_data)
    print('output_data.shape: {}'.format(output_data.shape))
