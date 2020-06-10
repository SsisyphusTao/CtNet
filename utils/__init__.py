from .ctdet import CTDetDataset
from .pascal import PascalVOC

def creat_dataset():
    class dataset(CTDetDataset, PascalVOC):
        pass
    return dataset