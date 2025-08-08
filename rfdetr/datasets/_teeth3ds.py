import os
import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes, box_convert
import rfdetr.datasets.transforms as T


from utils.color_utils import FDI2color, color2label, color2FDI


class Teeth3DSDetection(Dataset):
    def __init__(self, root, transforms, train_test_split = 0, is_train = True):
        self.root = Path(root)
        self.transforms = transforms

        self.images = []
        self.targets = []

        if train_test_split == 1:
            self.split_files = ['training_lower.txt', 'training_upper.txt'] if is_train else ['testing_lower.txt',
                                                                                         'testing_upper.txt']
        elif train_test_split == 2:
            self.split_files = ['public-training-set-1.txt', 'public-training-set-2.txt'] if is_train \
                else ['private-testing-set.txt']
        elif train_test_split == 0:
            self.split_files = ['training_lower_sample.txt', 'training_upper_sample.txt']
        else:
            raise ValueError(f'train_test_split should be 0, 1 or 2. not {train_test_split}')
    
    def _set_images_targets(self):
        for split_file in self.split_files:
            with open(os.path.join(self.root, 'split', split_file), 'r') as file:
                for line in tqdm(file, desc="Processing Patients"):
                    line = line.rstrip()
                    l_name = line.split('_')[0]
                    l_view = line.split('_')[1]

                    render_dir = os.path.join(self.root, 'sample', 'processed', l_view, l_name, 'render')
                    mask_dir = os.path.join(self.root, 'sample', 'processed', l_view, l_name, 'mask')

                    render_files = sorted(os.listdir(render_dir))
                    mask_files = sorted(os.listdir(mask_dir))

                    for render_file, mask_file in zip(render_files, mask_files):
                        render_path = os.path.join(render_dir, render_file)
                        mask_path = os.path.join(mask_dir, mask_file)

                        target = {}

                        img = Image.open(render_path).convert("RGB")
                        width, height = img.size
                        image_id = len(self.images) + 1

                        target["orig_size"] = torch.as_tensor([int(height), int(width)])
                        target["size"] = torch.as_tensor([int(height), int(width)])
                        target["image_id"] = image_id

                        boxes = []
                        classes = []
                        area = []
                        iscrowd = []

                        mask = np.array(Image.open(mask_path).convert("RGB"))
                        unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
                        instance_colors = [tuple(color) for color in unique_colors if tuple(color) in color2label]

                        for color in instance_colors:
                            if color not in color2label:
                                print(f"Warning: Color {color} not found in color2label. Skipping.")
                                continue

                            category_id = color2FDI[color]
                            if category_id == 0:
                                continue

                            binary_mask = (np.all(mask == np.array(color), axis=-1)).astype(np.uint8)
                            binary_mask_tensor = torch.tensor(binary_mask, dtype=torch.uint8).unsqueeze(0)
                            if binary_mask_tensor.sum() == 0:
                                print(f"Skipping empty mask for color {color}.")
                                continue
                            
                            box = masks_to_boxes(binary_mask_tensor).to(torch.float32)
                            boxes.append(box)
                            classes.append(category_id)
                            iscrowd.append(0)

                            box_wh = box_convert(box, in_fmt="xyxy", out_fmt="xywh")
                            for b in box_wh:
                                area.append(int(b[2] * b[3]))

                            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                            target["labels"] = torch.tensor(classes, dtype=torch.int64)
                            target["area"] = torch.tensor(area, dtype=torch.int64)
                            target["iscrowd"] = torch.tensor(iscrowd, dtype=torch.int64)

                        self.images.append(render_path)
                        self.targets.append(target)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        target = self.targets[idx]
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
