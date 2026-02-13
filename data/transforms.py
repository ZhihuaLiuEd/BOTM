import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomAffine
from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

class Resize:
    def __init__(self, out_size):
        self.out_size = (out_size, out_size) if isinstance(out_size, int) else out_size
    def __call__(self, *args):
        results = []
        for item in args:
            is_mask = False
            if isinstance(item, torch.Tensor) and item.dtype in [torch.long, torch.int64, torch.uint8, torch.int8]:
                is_mask = True
            elif isinstance(item, np.ndarray) and item.dtype in [np.int64, np.int32, np.uint8, np.int8]:
                is_mask = True
            elif isinstance(item, Image.Image) and item.mode == 'L':
                is_mask = True
            
            if isinstance(item, np.ndarray):
                item = torch.from_numpy(item).permute(2, 0, 1) if item.ndim == 3 else torch.from_numpy(item)
            
            interpolation = TF.InterpolationMode.NEAREST if is_mask else TF.InterpolationMode.BILINEAR
            resized = TF.resize(item, self.out_size, interpolation=interpolation)
            results.append(resized)
        return tuple(results)

class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob
    def __call__(self, *args):
        if random.random() < self.flip_prob:
            return tuple(TF.hflip(item) for item in args)
        return args

class RandomRotate:
    def __init__(self, angle_range=(-20, 20), rotate_prob=0.5):
        self.rotate_prob = rotate_prob
        self.angle_range = angle_range
    def __call__(self, *args):
        if random.random() < self.rotate_prob:
            angle = random.choice(self.angle_range)
            return tuple(TF.rotate(item, angle) for item in args)
        return args

class ToTensor:
    def __call__(self, *args):
        results = []
        for item in args:
            if isinstance(item, Image.Image):
                item = TF.to_tensor(item)
                results.append(item.float()) 
            elif isinstance(item, torch.Tensor):
                if item.dtype in [torch.int64, torch.int32, torch.uint8, torch.int8]:
                    results.append(item.long())
                else:
                    results.append(item.float())
            elif isinstance(item, np.ndarray):
                if item.dtype in [np.int64, np.int32, np.uint8, np.int8]:
                    results.append(torch.from_numpy(item).long())
                else:
                    results.append(torch.from_numpy(item).float())
            else:
                results.append(item)
        return tuple(results)

class Convert2PIL:
    def __call__(self, *args):
        results = []
        for item in args:
            if isinstance(item, np.ndarray):
                mode = 'L'
                if item.ndim == 3 and item.shape[2] == 3: mode = 'RGB'
                if item.ndim == 3 and item.shape[2] == 1: item = item[:, :, 0]
                
                if item.dtype in [np.uint8, np.int8, np.int64, np.int32]:
                    img = Image.fromarray(item.astype(np.uint8))
                else:
                    img = Image.fromarray((item * 255).astype(np.uint8) if item.max() <= 1 else item.astype(np.uint8))
                results.append(img)
            else:
                results.append(item)
        return tuple(results)