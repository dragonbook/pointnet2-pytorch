# Pointnet++ Part segmentation
This repo is implementation for [PointNet++](https://arxiv.org/abs/1706.02413) part segmentation model based on [PyTorch](https://pytorch.org) and [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric). 

**The model has been mergered into [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) as a point cloud segmentation [example](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py), you can try it.**

# Performance
Segmentation on  [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

| Method | mcIoU|Airplane|Bag|Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| PointNet++   | 81.9| 82.4| 79.0| 87.7| 77.3 |90.8| 71.8| 91.0| 85.9| 83.7| 95.3| 71.6| 94.1| 81.3| 58.7| 76.4| 82.6| 
| PointNet++(this repo) || 82.5| 76.1| 87.8| | | 73.7| | | | 95.3| 70.5


Note,
- mcIOU: mean per-class pIoU
- The model uses single-scale grouping with raw points as input.
- All experiments are trained with same default configration: npoints=2500, batchsize=8, num_epoches=30. The recorded accuracy above is the test accuracy of the final epoch.


# Requirements
- [PyTorch](https://pytorch.org)
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

# Sample segmentation result
![segmentation_result](figs/segmentation_result.png)


# Links
-  [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) by fxia22. This repo's tranining code is heavily borrowed from fxia22's repo.
- Official [PointNet](https://github.com/charlesq34/pointnet) and [PointNet++](https://github.com/charlesq34/pointnet2) tensorflow implementations
- [PointNet++ classification example](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py) of pytorch_geometric library
