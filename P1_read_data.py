from torch.utils.data import Dataset
from PIL import Image
import os

"""
Author  :hfli11
File    :P1_read_data.py
Project :PytorchTest
Time    :2023/1/10 10:22
Description 读取图片数据，根据文件夹已确定的label名，给图片标记label(MyData类)
"""

class MyData(Dataset):

    def __init__(self, root_dir, image_dir):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.path = os.path.join(self.root_dir, self.image_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        img = Image.open(img_item_path)
        label = self.image_dir.split("_")[0]
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
ants_image_dir = "ants_image"
bees_image_dir = "bees_image"
ants_dataset = MyData(root_dir, ants_image_dir)
bees_dataset = MyData(root_dir, bees_image_dir)

train_dataset = ants_dataset + bees_dataset
# 调用__getitem__方法
img, label = train_dataset[1]
print(label)
img.show()