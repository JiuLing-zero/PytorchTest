import cv2
import torchvision.datasets
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

"""
Author  :hfli11
File    :P5_UsefulTransforms.py
Project :PytorchTest
Time    :2023/1/10 10:22
Description transforms的一些方法的运用
"""

# 在Terminal里输入命令 : tensorboard --logdir=logs --port=6007
# 访问 http://localhost:6007/
# 如果writer写入了其他内容，直接刷新网页即可，网页会读取logs文件夹下的文件得到更新的内容

img_path = "dataset/train/bees_image/196658222_3fffd79c67.jpg"
img_pil = Image.open(img_path)
img = cv2.imread(img_path)

writer = SummaryWriter("logs")

# ToTensor 变为Tensor类型
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize 标准化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm.forward(img_tensor)
trans_norm1 = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm1 = trans_norm1.forward(img_tensor)
trans_norm2 = transforms.Normalize([0.5, 7, 3], [2, 0.1, 8])
img_norm2 = trans_norm2.forward(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)
writer.add_image("Normalize", img_norm1, 1)
writer.add_image("Normalize", img_norm2, 2)

# Resize 改变原大小
print(img_pil.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize.forward(img_pil)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2 改变原大小的第二种方式
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img_pil)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop(312)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img_pil)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()