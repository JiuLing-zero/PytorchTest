import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# tensorboard --logdir=logs --port=6007

img_path = "dataset/train/ants_image/0013035.jpg"
img_arr = cv2.imread(img_path)

writer = SummaryWriter("logs")

# 1、transforms该如何使用(python中)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img_arr)
# tensor_img是Tensor类型，包含一些机器学习需要的参数

writer.add_image("Tensor_img", tensor_img)

writer.close()
