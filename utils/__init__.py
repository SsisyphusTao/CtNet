from .ctdet import CTDetDataset
from .pascal import PascalVOC
from .coco import COCO
import torch

def get_dataset():
    class dataset(CTDetDataset, COCO):
        pass
    return dataset

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    inp = []
    hm = []
    reg_mask = []
    ind = []
    wh = []
    reg = []

    for sample in batch:
        inp.append(torch.from_numpy(sample[0]))
        hm.append(torch.from_numpy(sample[1]))
        reg_mask.append(torch.from_numpy(sample[2]))
        ind.append(torch.from_numpy(sample[3]))
        wh.append(torch.from_numpy(sample[4]))
        reg.append(torch.from_numpy(sample[5]))
    return {'input': torch.stack(inp,0),
            'hm': torch.stack(hm, 0),
            'reg_mask': torch.stack(reg_mask, 0),
            'ind': torch.stack(ind, 0),
            'wh': torch.stack(wh, 0),
            'reg': torch.stack(reg, 0)}

def val_collate(batch):
    inp = []
    meta = []
    for sample in batch:
        inp.append(sample[0])
        meta.append(sample[1])
    return {'input': torch.cat(inp,0),
        'meta': meta}