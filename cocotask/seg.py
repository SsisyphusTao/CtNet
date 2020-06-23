#%%
from pycocotools.coco import COCO
from pycocotools.mask import encode, decode, area, toBbox

import matplotlib.pyplot as plt
from PIL import Image
import random

coco = COCO('/root/data/coco/annotations/instances_val2017.json')
#%%
imgIds = coco.getImgIds()
imgIds = imgIds[random.randint(0, len(imgIds))]

file_name = coco.loadImgs(ids=imgIds)[0]['file_name']
annIds = coco.getAnnIds(imgIds=imgIds)
ann = coco.loadAnns(annIds)

# mask=coco.annToMask(ann)
# rle=coco.annToRLE(ann)

# rle=encode(mask)
# mask=decode(rle)

# area(rle)
# toBbox(rle)

plt.imshow(Image.open('/root/data/coco/val2017/'+file_name))
plt.axis('off')
coco.showAnns(ann)
# %%
