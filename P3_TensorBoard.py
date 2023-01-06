from torch.utils.tensorboard import SummaryWriter
import cv2

# tensorboard --logdir=logs --port=6007

writer = SummaryWriter("logs")

img_path = "dataset/train/bees_image/16838648_415acd9e3f.jpg"
img_array = cv2.imread(img_path)
print(type(img_array))
print(img_array.shape)
writer.add_image("tess", img_array, 2, dataformats='HWC')

# y = x*x
for i in range(100):
    writer.add_scalar("y=x*x", i*i, i)

writer.close()