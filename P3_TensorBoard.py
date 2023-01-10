from torch.utils.tensorboard import SummaryWriter
import cv2

"""
Author  :hfli11
File    :P3_TensorBoard.py
Project :PytorchTest
Time    :2023/1/10 10:22
Description TensorBoard的使用，加载一个图片和一个数学函数图
"""

# 在Terminal里输入命令 : tensorboard --logdir=logs --port=6007
# 访问 http://localhost:6007/
# 如果writer写入了其他内容，直接刷新网页即可，网页会读取logs文件夹下的文件得到更新的内容

# 设置logs文件夹，存储写入的内容，以供tensorboard网页端解析使用
writer = SummaryWriter("logs")

# 读取一个文件夹然后通过cv2,转换格式
img_path = "dataset/train/bees_image/16838648_415acd9e3f.jpg"
img_array = cv2.imread(img_path)
print(type(img_array))# ps:numpy.ndarray
print(img_array.shape)# ps:(450, 500, 3)
# 写入一个tag为"tess",图片为img_array,step步数为2
# dataformats(设置img_tensor,通过查看img_array.shape确认为'HWC',具体看add_image原码)
writer.add_image("tess", img_array, 2, dataformats='HWC')

# y = x*x
for i in range(100):
    writer.add_scalar("y=x*x", i*i, i)

writer.close()