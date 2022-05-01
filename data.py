import cv2
from torch.utils.data import Dataset
from torchvision.io import read_image
from glob import glob
import os

class ClasslessImageDataset(Dataset):
    def __init__(self, image_dir, split="train", transform=None, target_transform=None):
        assert split in ["train", "test"]
        self.img_dir = os.path.join(image_dir, split)
        self.transform = transform
        self.target_transform = target_transform
        self.images = glob(self.img_dir + "/*.*")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = read_image(img_path)
        image_out = image
        target_out = image
        if self.transform:
            image_out = self.transform(image_out)
        if self.target_transform:
            target_out = self.target_transform(target_out)
        return image_out, target_out

    
class ClasslessImageDatasetAlbCv(Dataset):
    def __init__(self, image_dir, split="train", transform=None, target_transform=None):
        assert split in ["train", "test"]
        self.img_dir = os.path.join(image_dir, split)
        self.transform = transform
        self.target_transform = target_transform
        self.images = glob(self.img_dir + "/*.*")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_out = image
        target_out = image
        if self.transform:
            image_out = self.transform(image=image)["image"]
        if self.target_transform:
            target_out = self.target_transform(image=image)["image"]
        return image_out, target_out