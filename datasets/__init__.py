import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset, PDBPairDataset


def get_dataset(config, *args, **kwargs):
    # name = config.name
    root = config.path
    
    dataset = PocketLigandPairDataset(root, *args, **kwargs)
    
    
    # if name == 'pl':
    #     dataset = PocketLigandPairDataset(root, *args, **kwargs)
    # elif name == 'pdbbind':
    #     dataset = PDBPairDataset(root, *args, **kwargs)
    # else:
    #     raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
