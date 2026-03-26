from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

class WorldGhibliDataset(Dataset):
    def __init__(self, root_world, root_ghibli, transform=None):
        super().__init__()
        self.root_world = root_world
        self.root_ghibli = root_ghibli
        self.transform = transform

        self.world_image = os.listdir(root_world)
        self.ghibli_image = os.listdir(root_ghibli)
        self.length_dataset = max(len(self.world_image),len(self.ghibli_image))
        self.world_len = len(self.world_image)
        self.ghibli_len = len(self.ghibli_image)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        ghibli_img = self.ghibli_image[index % self.ghibli_len]
        world_img = self.world_image[index % self.world_len]

        ghibli_path = os.path.join(self.root_ghibli, ghibli_img)
        world_path = os.path.join(self.root_world, world_img)

        ghibli_img = np.array(Image.open(ghibli_path).convert("RGB"))
        world_img = np.array(Image.open(world_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=ghibli_img, image0=world_img)
            ghibli_img = augmentations["image"]
            world_img = augmentations["image0"]

        return ghibli_img, world_img