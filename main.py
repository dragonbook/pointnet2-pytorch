'''
Modified from https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_segmentation.py
'''

import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
import torch.backends.cudnn as cudnn


from dataset.shapenet import ShapeNetPartSegDataset
from model.pointnet2_part_seg import PointNet2PartSegmentNet
import torch_geometric.transforms as GT

import time


## Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='shapenet', help='dataset path')
parser.add_argument('--category', type=str, default='Airplane', help='select category')
parser.add_argument('--npoints', type=int, default=2500, help='resample points number')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='checkpoint', help='output folder')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--test_per_batches', type=int, default=10, help='run a test batch per training batches number')
parser.add_argument('--num_workers', type=int, default=6, help='number of data loading workers')

opt = parser.parse_args()
print(opt)


## Random seed
# opt.manual_seed = np.random.randint(1, 10000)  # fix seed
# TODO: Still cannot get determinstic result
opt.manual_seed = 123
print('Random seed: ', opt.manual_seed)
random.seed(opt.manual_seed)
np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
torch.cuda.manual_seed(opt.manual_seed)


## Dataset and transform
print('Construct dataset ..')
rot_max_angle = 15
trans_max_distance = 0.01

RotTransform = GT.Compose([GT.RandomRotate(rot_max_angle, 0), GT.RandomRotate(rot_max_angle, 1), GT.RandomRotate(rot_max_angle, 2)])
TransTransform = GT.RandomTranslate(trans_max_distance)

train_transform = GT.Compose([GT.NormalizeScale(), RotTransform, TransTransform])
test_transform = GT.Compose([GT.NormalizeScale(), ])

dataset = ShapeNetPartSegDataset(
    root_dir=opt.dataset, category=opt.category, train=True, transform=train_transform, npoints=opt.npoints)  
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

test_dataset = ShapeNetPartSegDataset(
    root_dir=opt.dataset, category=opt.category, train=False, transform=test_transform, npoints=opt.npoints)
# Note, set shuffle=True for peridodic running a random test batch during training
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

num_classes = dataset.num_classes()

print('dataset size: ', len(dataset))
print('test_dataset size: ', len(test_dataset))
print('num_classes: ', num_classes)

try:
    os.mkdir(opt.outf)
except OSError:
    pass


## Model, criterion and optimizer
print('Construct model ..')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float
print('cudnn.enabled: ', torch.backends.cudnn.enabled)


net = PointNet2PartSegmentNet(num_classes)

if opt.model != '':
    net.load_state_dict(torch.load(opt.model))
net = net.to(device, dtype)

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters())


## Train
print('Training ..')
blue = lambda x: '\033[94m' + x + '\033[0m'
num_batch = len(dataset) // opt.batch_size
test_per_batches = opt.test_per_batches

print('number of epoches: ', opt.nepoch)
print('number of batches per epoch: ', num_batch)
print('run test per batches: ', test_per_batches)

for epoch in range(opt.nepoch):
    print('Epoch {}, total epoches {}'.format(epoch+1, opt.nepoch))

    net.train()

    for batch_idx, sample in enumerate(dataloader):
        # points: (batch_size, n, 3)
        # labels: (batch_size, n)
        points, labels = sample['points'], sample['labels']
        points = points.transpose(1, 2).contiguous()  # (batch_size, 3, n)
        points, labels = points.to(device, dtype), labels.to(device, torch.long)

        optimizer.zero_grad()

        pred = net(points)  # (batch_size, n, num_classes)
        pred = pred.view(-1, num_classes)  # (batch_size * n, num_classes) 
        target = labels.view(-1, 1)[:, 0]

        loss = F.nll_loss(pred, target)
        loss.backward()

        optimizer.step()

        ##
        pred_label = pred.detach().max(1)[1] 
        correct = pred_label.eq(target.detach()).cpu().sum()
        total = pred_label.shape[0]

        print('[{}: {}/{}] train loss: {} accuracy: {}'.format(epoch, batch_idx, num_batch, loss.item(), float(correct.item())/total))

        ##
        if batch_idx % test_per_batches == 0:
            print('Run a test batch')
            net.eval()

            with torch.no_grad():
                batch_idx, sample = next(enumerate(test_dataloader))

                points, labels = sample['points'], sample['labels']
                points = points.transpose(1, 2).contiguous()
                points, labels = points.to(device, dtype), labels.to(device, torch.long)

                pred = net(points)
                pred = pred.view(-1, num_classes)
                target = labels.view(-1, 1)[:, 0]

                loss = F.nll_loss(pred, target)

                pred_label = pred.detach().max(1)[1]
                correct = pred_label.eq(target.detach()).cpu().sum()
                total = pred_label.shape[0]
                print('[{}: {}/{}] {} loss: {} accuracy: {}'.format(epoch, batch_idx, num_batch, blue('test'), loss.item(), float(correct.item())/total))

            # Back to training mode
            net.train()
                
    torch.save(net.state_dict(), '{}/seg_model_{}_{}.pth'.format(opt.outf, opt.category, epoch))


## Benchmarm mIOU
net.eval()
shape_ious = []

with torch.no_grad():
    for batch_idx, sample in enumerate(test_dataloader):
        points, labels = sample['points'], sample['labels']
        points = points.transpose(1, 2).contiguous()
        points = points.to(device, dtype)

        # start_t = time.time()
        pred = net(points) # (batch_size, n, num_classes)
        # print('batch inference forward time used: {} ms'.format(time.time() - start_t))

        pred_label = pred.max(2)[1]
        pred_label = pred_label.cpu().numpy()
        target_label = labels.numpy()

        batch_size = target_label.shape[0]
        for shape_idx in range(batch_size):
            parts = range(num_classes)  # np.unique(target_label[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_label[shape_idx] == part, target_label[shape_idx] == part))
                U = np.sum(np.logical_or(pred_label[shape_idx] == part, target_label[shape_idx] == part))
                if U == 0: iou = 1
                else: iou = float(I) / U
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print('mIOU for category {}: {}'.format(opt.category, np.mean(shape_ious)))

print('Done.')
