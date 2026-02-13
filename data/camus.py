import os
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from .transforms import Compose, Resize, ToTensor, Convert2PIL, RandomHorizontalFlip, RandomRotate

class CAMUSDataset(Dataset):
    def __init__(self, root, view='2CH', transforms=None, return_indices=True, split='train', val_ratio=0.1):
        super().__init__()
        self.root = root
        self.view = view.upper()
        self.transforms = transforms
        self.return_indices = return_indices
        
        if not os.path.exists(self.root):
            raise ValueError(f"Dataset root not found: {self.root}")
            
        all_patients = sorted([p for p in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, p))])
        
        if split == 'test':
            self.patient_list = all_patients
        else:
            random.seed(42) 
            random.shuffle(all_patients)
            val_size = int(len(all_patients) * val_ratio)
            
            if split == 'train':
                self.patient_list = all_patients[val_size:]
            elif split == 'val':
                self.patient_list = all_patients[:val_size]
            else:
                raise ValueError(f"Unknown split: {split}")

    def __getitem__(self, index):
        patient = self.patient_list[index]
        patient_path = os.path.join(self.root, patient)
        
        paths = {
            'ES': os.path.join(patient_path, f"{patient}_{self.view}_ES.mhd"),
            'ED': os.path.join(patient_path, f"{patient}_{self.view}_ED.mhd"),
            'ES_gt': os.path.join(patient_path, f"{patient}_{self.view}_ES_gt.mhd"),
            'ED_gt': os.path.join(patient_path, f"{patient}_{self.view}_ED_gt.mhd"),
        }
        
        data = {}
        for key, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img) 
            data[key] = arr[0]

        es_img = np.expand_dims(data['ES'], axis=-1).astype(np.float32)
        ed_img = np.expand_dims(data['ED'], axis=-1).astype(np.float32)
        es_label = data['ES_gt'].astype(np.uint8)
        ed_label = data['ED_gt'].astype(np.uint8)

        if self.transforms is not None:
            es_img, es_label, ed_img, ed_label = self.transforms(es_img, es_label, ed_img, ed_label)
        else:
            es_img = torch.from_numpy(es_img).permute(2,0,1).float()
            ed_img = torch.from_numpy(ed_img).permute(2,0,1).float()
            es_label = torch.from_numpy(es_label).long()
            ed_label = torch.from_numpy(ed_label).long()

        if self.return_indices:
            if isinstance(es_label, torch.Tensor) and es_label.ndim == 3:
                es_label = es_label.squeeze(0).long()
                ed_label = ed_label.squeeze(0).long()
            elif isinstance(es_label, np.ndarray):
                es_label = torch.from_numpy(es_label).long()
                ed_label = torch.from_numpy(ed_label).long()
        else:
            if not isinstance(es_label, torch.Tensor):
                es_label = torch.from_numpy(es_label).long()
                ed_label = torch.from_numpy(ed_label).long()
            es_label = torch.nn.functional.one_hot(es_label, num_classes=4).permute(2, 0, 1).float()
            ed_label = torch.nn.functional.one_hot(ed_label, num_classes=4).permute(2, 0, 1).float()

        return es_img, es_label, ed_img, ed_label

    def __len__(self):
        return len(self.patient_list)

def get_camus_transform(is_train=True, out_size=672):
    t_list = [Convert2PIL()]
    if is_train:
        t_list.extend([
            RandomHorizontalFlip(),
            RandomRotate()
        ])
    t_list.extend([
        Resize(out_size),
        ToTensor()
    ])
    return Compose(t_list)