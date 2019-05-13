# Pointnet2 Part segmentation
This repo is implementation for [PointNet++](https://arxiv.org/abs/1706.02413) part segmentation model in PyTorch based on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric). It can achieve comparable or better performance even compared with [PointCNN](https://arxiv.org/abs/1801.07791) on Shapenet dataset.


# Requirements
- PyTorch
- [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
- [Open3D](https://github.com/intel-isl/Open3D)(optional, for visualization of segmentation result)

## Quickly install pytorch_geometric and Open3D with Anaconda
```
$ pip install --verbose --no-cache-dir torch-scatter
$ pip install --verbose --no-cache-dir torch-sparse
$ pip install --verbose --no-cache-dir torch-cluster
$ pip install --verbose --no-cache-dir torch-spline-conv (optional)
$ pip install torch-geometric
```

```
# optional
conda install -c open3d-admin open3d
```

# Usage
Training
```
python main.py
```

Show segmentation result
```
python vis/show_seg_res.py
```

# Performance
Segmentation on  [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

| Class(mIOU) |Airplane|Bag|Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| PointNet++   | 82.4| 79.0| 87.7| 77.3 |90.8| 71.8| 91.0| 85.9| 83.7| 95.3| 71.6| 94.1| 81.3| 58.7| 76.4| 82.6| 
| PointCNN     | 84.11| 86.47| 86.04| 80.83| 90.62| 79.70| 92.32| 88.44| 85.31| 96.11| **77.20**| 95.28| 84.21| 64.23| **80.00**| 82.99| 
| PointNet++(this repo) | | | | | | | | **88.56**| **85.72**| **97.00**| 72.94| **96.88**| **84.52**| **64.38**| 79.39| **85.91**|

<!-- | PointNet Offical     | 83.4| 78.7| 82.5| 74.9 |89.6| 73.0| 91.5| 85.9| 80.8| 95.3| 65.2| 93.0| 81.2| 57.9| 72.8| 80.6|  -->
<!-- | PointNet this repo   | 82.5| 79.6| 79.4| 71.6| 89.9| 72.5| 90.0| 86.1| 80.3| 96.3| 57.4| 91.2| 83.0| 60.3| 65.4| 86.0| -->
<!-- | PointNet++ this repo(w/o bn) | 84.8| 80.9| 86.3| 75.5| 90.6| 71.2| 90.5| 87.4| 83.0| 96.7| 58.0| 96.1| 82.5| 55.6| 72.2| 84.6| -->

<!-- Note that, -->
<!-- - This implementation trains each class separately -->
<!-- - There are some minimal implemention differences compared with offical repo -->
<!-- - Some default used training configurations: batch_size=8, nepochs=25, optimizer=adam -->


Sample segmentation result:
![segmentation_result](figs/segmentation_result.png)


# Links
-  [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) by fxia22. This repo's PointNet and tranining code are heavily borrowed from fxia22's repo.
- Official [PointNet](https://github.com/charlesq34/pointnet) and [PointNet++](https://github.com/charlesq34/pointnet2) tensorflow implementations
- [PointNet++ classification example](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet%2B%2B.py) of pytorch_geometric library
