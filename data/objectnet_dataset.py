import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

torch.manual_seed(0)

NORMALIZE = Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

DEFAULT_TRANSFORM = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    NORMALIZE,
])


class ObjectNetDataset(Dataset):
    def __init__(self, imagenet_path):
        self._imagenet_path = imagenet_path
        self._all_images = []
        self._tag_list = []

        objectnet_dataset = ImageFolder(self._imagenet_path)
        mappings_folder = os.path.abspath(os.path.join(self._imagenet_path, "../mappings"))

        objectnet_idx_to_imagenet_idxs = self._build_class_mappings(
            mappings_folder, objectnet_dataset.class_to_idx
        )

        for filepath, objectnet_idx in objectnet_dataset.samples:
            if objectnet_idx not in objectnet_idx_to_imagenet_idxs:
                continue

            relative_path = os.path.relpath(filepath, self._imagenet_path)
            primary_imagenet_idx = objectnet_idx_to_imagenet_idxs[objectnet_idx][0]

            if primary_imagenet_idx not in self._tag_list:
                self._tag_list.append(primary_imagenet_idx)

            self._all_images.append((relative_path, primary_imagenet_idx))

    def _build_class_mappings(self, mappings_folder, folder_to_objectnet_idx):
        with open(os.path.join(mappings_folder, "objectnet_to_imagenet_1k.json")) as f:
            raw_objectnet_to_imagenet = json.load(f)

        objectnet_to_imagenet_labels = {
            obj_label: img_labels.split("; ")
            for obj_label, img_labels in raw_objectnet_to_imagenet.items()
        }

        with open(os.path.join(mappings_folder, "folder_to_objectnet_label.json")) as f:
            folder_to_objectnet_label = json.load(f)

        objectnet_label_to_idx = {
            obj_label: folder_to_objectnet_idx[folder]
            for folder, obj_label in folder_to_objectnet_label.items()
        }

        with open(os.path.join(mappings_folder, "pytorch_to_imagenet_2012_id.json")) as f:
            imagenet_idx_to_line = json.load(f)

        with open(os.path.join(mappings_folder, "imagenet_to_label_2012_v2")) as f:
            raw_line_to_imagenet_label = f.readlines()

        line_to_imagenet_label = {
            line_idx: label.strip()
            for line_idx, label in enumerate(raw_line_to_imagenet_label)
        }

        imagenet_label_to_idx = {
            line_to_imagenet_label[line_idx]: int(img_idx)
            for img_idx, line_idx in imagenet_idx_to_line.items()
        }

        objectnet_idx_to_imagenet_idxs = {
            objectnet_label_to_idx[obj_label]: [
                imagenet_label_to_idx[img_label] for img_label in img_labels
            ]
            for obj_label, img_labels in objectnet_to_imagenet_labels.items()
        }

        return objectnet_idx_to_imagenet_idxs

    def __getitem__(self, index):
        image_path, classification = self._all_images[index]
        full_image_path = os.path.join(self._imagenet_path, image_path)

        image = Image.open(full_image_path).convert('RGB')
        image = DEFAULT_TRANSFORM(image)

        return image, classification

    def __len__(self):
        return len(self._all_images)