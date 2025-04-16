# from .kitti_dataset import KITTIDataset
from .scared_dataset import ScaredDataset
from .sceneflow_dataset import SceneFlowDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "scared": ScaredDataset
    # "kitti": KITTIDataset
}
