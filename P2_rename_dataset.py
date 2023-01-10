import os

"""
Author  :hfli11
File    :P2_rename_dataset.py
Project :PytorchTest
Time    :2023/1/10 10:22
Description 给以label命名的图片文件夹中的图片，建立对应的label标签文件夹
"""

def rename(target_dir):
    root_dir = "dataset/train"
    img_path = os.listdir(os.path.join(root_dir, target_dir))
    label = target_dir.split("_")[0]
    out_dir = label + "_label"
    for i in img_path:
        file_name = i.split(".jpg")[0]
        # 存储与图片名称相同的.txt文件,内容为该图片的label标签
        with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
            f.write(label)


target_dir1 = "ants_image"
target_dir2 = "bees_image"
rename(target_dir1)
rename(target_dir2)

