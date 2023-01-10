import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

"""
Author  :hfli11
File    :P4_Transforms.py
Project :PytorchTest
Time    :2023/1/10 10:22
Description transforms.ToTensor()的简单练习
"""

# 在Terminal里输入命令 : tensorboard --logdir=logs --port=6007
# 访问 http://localhost:6007/
# 如果writer写入了其他内容，直接刷新网页即可，网页会读取logs文件夹下的文件得到更新的内容

img_path = "dataset/train/ants_image/0013035.jpg"
img_arr = cv2.imread(img_path)

writer = SummaryWriter("logs")

# 1、transforms该如何使用(python中)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img_arr)
# tensor_img是Tensor类型，包含一些机器学习需要的参数

writer.add_image("Tensor_img", tensor_img)

writer.close()
