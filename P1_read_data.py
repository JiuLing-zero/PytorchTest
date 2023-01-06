from torch.utils.data import Dataset
from PIL import Image
import os


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


def main():
    root_dir = "dataset/train"
    ants_image_dir = "ants_image"
    bees_image_dir = "bees_image"
    ants_dataset = MyData(root_dir, ants_image_dir)
    bees_dataset = MyData(root_dir, bees_image_dir)

    train_dataset = ants_dataset + bees_dataset

    img, label = train_dataset[1]
    print(label)
    img.show()


if __name__ == '__main__':
    main()
