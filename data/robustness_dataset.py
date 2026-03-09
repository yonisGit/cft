import json
import os
from collections import namedtuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from label_str_to_imagenet_classes import label_str_to_imagenet_classes

torch.manual_seed(0)

ImageItem = namedtuple('ImageItem', ['image_name', 'tag'])

NORMALIZE = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORMALIZE,
])


class RobustnessDataset(Dataset):
    def __init__(self, imagenet_path, imagenet_classes_path='imagenet_classes.json', is_v2=False, is_si=False):
        self._is_v2 = is_v2
        self._is_si = is_si
        self._imagenet_path = imagenet_path

        with open(imagenet_classes_path, 'r') as f:
            self._imagenet_classes = json.load(f)

        self._all_images = []
        tag_list = os.listdir(self._imagenet_path)

        for tag in tag_list:
            base_dir = os.path.join(self._imagenet_path, tag)
            for file_name in os.listdir(base_dir):
                self._all_images.append(ImageItem(file_name, tag))

    def __len__(self):
        return len(self._all_images)

    def __getitem__(self, index):
        image_item = self._all_images[index]
        image_path = os.path.join(self._imagenet_path, image_item.tag, image_item.image_name)

        image = Image.open(image_path).convert('RGB')
        image = DEFAULT_TRANSFORM(image)

        if self._is_v2:
            class_label = int(image_item.tag)
        elif self._is_si:
            class_label = int(label_str_to_imagenet_classes[image_item.tag])
        else:
            class_label = int(self._imagenet_classes[image_item.tag])

        return image, class_label