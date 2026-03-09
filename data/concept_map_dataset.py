import json
import os
import random
from collections import namedtuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

torch.manual_seed(0)

SegItem = namedtuple('SegItem', ['image_name', 'tag'])

NORMALIZE = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE
])

TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

MERGED_TAGS = {
    'n04356056', 'n04355933', 'n04493381', 'n02808440', 'n03642806',
    'n03832673', 'n04008634', 'n03773504', 'n03887697', 'n15075141'
}

TRAIN_PARTITION = "train"
VAL_PARTITION = "val"
LEGAL_PARTITIONS = {TRAIN_PARTITION, VAL_PARTITION}


class ConceptMapDataset(Dataset):
    def __init__(self, seg_path, imagenet_path, partition=TRAIN_PARTITION,
                 num_samples=2, train_classes=500,
                 imagenet_classes_path='imagenet_classes.json', seed=None):

        if partition not in LEGAL_PARTITIONS:
            raise ValueError(f"Unsupported partition type: {partition}")

        self._partition = partition
        self._seg_path = seg_path
        self._imagenet_path = imagenet_path

        with open(imagenet_classes_path, 'r') as f:
            self._imagenet_classes = json.load(f)

        self._tag_list = [tag for tag in os.listdir(self._seg_path) if tag not in MERGED_TAGS]

        if seed is not None:
            random.seed(seed)
            random.shuffle(self._tag_list)

        if partition == TRAIN_PARTITION:
            self._tag_list = self._tag_list[:train_classes]
        elif partition == VAL_PARTITION:
            self._tag_list = self._tag_list[train_classes:]

        for tag in self._tag_list:
            if tag not in self._imagenet_classes:
                raise KeyError(f"Tag {tag} not found in ImageNet classes.")

        self._segmentation_items = []
        for tag in self._tag_list:
            base_dir = os.path.join(self._seg_path, tag)
            for i, seg in enumerate(os.listdir(base_dir)):
                if i >= num_samples:
                    break
                image_name = seg.split('.')[0]
                self._segmentation_items.append(SegItem(image_name, tag))

    def __len__(self):
        return len(self._segmentation_items)

    def __getitem__(self, index):
        seg_item = self._segmentation_items[index]

        seg_path = os.path.join(self._seg_path, seg_item.tag, f"{seg_item.image_name}.png")
        image_path = os.path.join(self._imagenet_path, seg_item.tag, f"{seg_item.image_name}.JPEG")

        concept_map_img = Image.open(seg_path)
        image = Image.open(image_path).convert('RGB')

        concept_map_np = np.array(concept_map_img)
        concept_map_np = concept_map_np[:, :, 1] * 256 + concept_map_np[:, :, 0]

        concept_map_np[concept_map_np == 1000] = 0
        concept_map_np[concept_map_np != 0] = 1

        seg_tensor = torch.from_numpy(concept_map_np.astype(np.float32))
        seg_tensor = seg_tensor.unsqueeze(0)

        image, seg_tensor = self._apply_transforms(image, seg_tensor)
        image_tensor = IMAGE_TRANSFORMS(image)
        class_name = int(self._imagenet_classes[seg_item.tag])

        return seg_tensor, image_tensor, class_name

    def _apply_transforms(self, image, concept_map):
        if self._partition == VAL_PARTITION:
            image = TRANSFORM_EVAL(image)
            concept_map = TRANSFORM_EVAL(concept_map)

        elif self._partition == TRAIN_PARTITION:
            image = TF.resize(image, size=[256, 256])
            concept_map = TF.resize(concept_map, size=[256, 256])

            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
            image = TF.crop(image, i, j, h, w)
            concept_map = TF.crop(concept_map, i, j, h, w)

            if random.random() > 0.5:
                image = TF.hflip(image)
                concept_map = TF.hflip(concept_map)

        return image, concept_map