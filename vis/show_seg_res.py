# Warning: import open3d may lead crash, try to import open3d first here
from view import view_points_labels

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')  # add project root directory

from dataset.shapenet import ShapeNetPartSegDataset
from model.pointnet2_part_seg import PointNet2PartSegmentNet
import torch_geometric.transforms as GT
import torch
import numpy as np
import argparse


##
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='shapenet', help='dataset path')
parser.add_argument('--category', type=str, default='Airplane', help='select category')
parser.add_argument('--npoints', type=int, default=2500, help='resample points number')
parser.add_argument('--model', type=str, default='./checkpoint/seg_model_Airplane_24.pth', help='model path')
parser.add_argument('--sample_idx', type=int, default=0, help='select a sample to segment and view result')

opt = parser.parse_args()
print(opt)


## Load dataset
print('Construct dataset ..')
test_transform = GT.Compose([GT.NormalizeScale(),])

test_dataset = ShapeNetPartSegDataset(
    root_dir=opt.dataset,
    category=opt.category,
    train=False,
    transform=test_transform,
    npoints=opt.npoints
)
num_classes = test_dataset.num_classes()

print('test dataset size: ', len(test_dataset))
print('num_classes: ', num_classes)


# Load model
print('Construct model ..')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

# net = PointNetPartSegmentNet(num_classes)
net = PointNet2PartSegmentNet(num_classes)

net.load_state_dict(torch.load(opt.model))
net = net.to(device, dtype)
net.eval()


##
def eval_sample(net, sample):
    '''
    sample: { 'points': tensor(n, 3), 'labels': tensor(n,) }
    return: (pred_label, gt_label) with labels shape (n,)
    '''
    net.eval()
    with torch.no_grad():
        # points: (n, 3)
        points, gt_label = sample['points'], sample['labels']
        n = points.shape[0]

        points = points.view(1, n, 3)  # make a batch
        points = points.transpose(1, 2).contiguous()
        points = points.to(device, dtype)

        pred = net(points)  # (batch_size, n, num_classes)
        pred_label = pred.max(2)[1]
        pred_label = pred_label.view(-1).cpu()  # (n,)

        assert pred_label.shape == gt_label.shape
        return (pred_label, gt_label)
        

def compute_mIoU(pred_label, gt_label):
    minl, maxl = np.min(gt_label), np.max(gt_label)
    ious = []
    for l in range(minl, maxl+1):
        I = np.sum(np.logical_and(pred_label == l, gt_label == l))
        U = np.sum(np.logical_or(pred_label == l, gt_label == l))
        if U == 0: iou = 1 
        else: iou = float(I) / U
        ious.append(iou)
    return np.mean(ious)


def label_diff(pred_label, gt_label):
    '''
    Assign 1 if different label, or 0 if same label  
    '''
    diff = pred_label - gt_label
    diff_mask = (diff != 0)

    diff_label = np.zeros((pred_label.shape[0]), dtype=np.int32)
    diff_label[diff_mask] = 1

    return diff_label


# Get one sample and eval
sample = test_dataset[opt.sample_idx]

print('Eval test sample ..')
pred_label, gt_label = eval_sample(net, sample)
print('Eval done ..')


# Get sample result
print('Compute mIoU ..')
points = sample['points'].numpy()
pred_labels = pred_label.numpy()
gt_labels = gt_label.numpy()
diff_labels = label_diff(pred_labels, gt_labels)

print('mIoU: ', compute_mIoU(pred_labels, gt_labels))


# View result

# print('View gt labels ..')
# view_points_labels(points, gt_labels)

# print('View diff labels ..')
# view_points_labels(points, diff_labels)

print('View pred labels ..')
view_points_labels(points, pred_labels)
