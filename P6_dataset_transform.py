import torchvision
from torch.utils.tensorboard import SummaryWriter

"""
Author  :hfli11
File    :P6_dataset_transform.py
Project :PytorchTest
Time    :2023/1/10 10:22
Description torchvision
"""

# 在Terminal里输入命令 : tensorboard --logdir=p10 --port=6007
# 访问 http://localhost:6007/
# 如果writer写入了其他内容，直接刷新网页即可，网页会读取logs文件夹下的文件得到更新的内容

# 访问 https://pytorch.org/vision/stable/index.html
# 可以查看一些Datasets数据集 以及 模型和预先训练的权重(Models and pre-trained weights)

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=dataset_transform, download=True)

# 在数据集获取时不加transform，则下面语句可用(因为没有转为tensor格式)
# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()