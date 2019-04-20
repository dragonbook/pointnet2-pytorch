import open3d as o3d
import numpy as np


def mini_color_table(index, norm=True):
    colors = [
        [0.5000, 0.5400, 0.5300], [0.8900, 0.1500, 0.2100], [0.6400, 0.5800, 0.5000],
        [1.0000, 0.3800, 0.0100], [1.0000, 0.6600, 0.1400], [0.4980, 1.0000, 0.0000],
        [0.4980, 1.0000, 0.8314], [0.9412, 0.9725, 1.0000], [0.5412, 0.1686, 0.8863],
        [0.5765, 0.4392, 0.8588], [0.3600, 0.1400, 0.4300], [0.5600, 0.3700, 0.6000],
    ]

    assert index >= 0 and index < len(colors)
    color = colors[index]

    if not norm:
        color[0] *= 255
        color[1] *= 255
        color[2] *= 255

    return color


def view_points(points, colors=None):
    '''
    points: np.ndarray with shape (n, 3)
    colors: [r, g, b] or np.array with shape (n, 3)
    '''
    cloud = o3d.PointCloud()
    cloud.points = o3d.Vector3dVector(points) 

    if colors is not None:
        if isinstance(colors, np.ndarray):
            cloud.colors = o3d.Vector3dVector(colors)
        else: cloud.paint_uniform_color(colors)

    o3d.draw_geometries([cloud])


def label2color(labels):
    '''
    labels: np.ndarray with shape (n, )
    colors(return): np.ndarray with shape (n, 3)
    '''
    num = labels.shape[0]
    colors = np.zeros((num, 3))

    minl, maxl = np.min(labels), np.max(labels)
    for l in range(minl, maxl + 1):
        colors[labels==l, :] = mini_color_table(l) 

    return colors


def view_points_labels(points, labels):
    '''
    Assign points with colors by labels and view colored points.
    points: np.ndarray with shape (n, 3)
    labels: np.ndarray with shape (n, 1), dtype=np.int32
    '''
    assert points.shape[0] == labels.shape[0]
    colors = label2color(labels)
    view_points(points, colors)
