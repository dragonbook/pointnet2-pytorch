import os
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.datasets import ShapeNet


class ShapeNetPartSegDataset(Dataset):
    '''
    Resample raw point cloud to fixed number of points.
    Map raw label from range [1, N] to [0, N-1].
    '''
    def __init__(self, root_dir, category, train=True, transform=None, npoints=2500):
        categories = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
            'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']
        # assert os.path.exists(root_dir) 
        assert category in categories

        self.npoints = npoints
        self.dataset = ShapeNet(root_dir, category, train, transform)

    def __getitem__(self, index):
        data = self.dataset[index]
        points, labels = data.pos, data.y
        assert labels.min() >= 0

        # Resample to fixed number of points
        choice = np.random.choice(points.shape[0], self.npoints, replace=True)
        points, labels = points[choice, :], labels[choice]

        sample = {
            'points': points,  # torch.Tensor (n, 3)
            'labels': labels   # torch.Tensor (n,)
        }

        return sample

    def __len__(self):
        return len(self.dataset)

    def num_classes(self):
        return self.dataset.num_classes
