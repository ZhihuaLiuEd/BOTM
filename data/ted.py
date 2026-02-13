import os
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from .transforms import Compose, Resize, ToTensor

class TEDDataset(Dataset):
    def __init__(self, root, patient_list=None, transforms=None, split='train', val_ratio=0.1):
        super().__init__()
        self.root = root
        self.transforms = transforms

        if patient_list is None:
            all_patients = sorted(os.listdir(root))
            
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
            self.patient_list = patient_list

    def __getitem__(self, index):
        patient = self.patient_list[index]
        patient_path = os.path.join(self.root, patient)
        
        video_path = os.path.join(patient_path, f"{patient}_4CH_sequence.mhd")
        label_path = os.path.join(patient_path, f"{patient}_4CH_sequence_gt.mhd")

        video_sitk = sitk.ReadImage(video_path, sitk.sitkFloat32)
        label_sitk = sitk.ReadImage(label_path, sitk.sitkInt8)
        
        video_arr = sitk.GetArrayFromImage(video_sitk)
        label_arr = sitk.GetArrayFromImage(label_sitk)

        frame_pairs, labels = self.generate_pair(video_arr, label_arr)

        return frame_pairs, labels

    def generate_pair(self, video_arr, label_arr):
        num_frames, H, W = video_arr.shape
        frame_pairs = []
        labels = []
        
        for t in range(num_frames):
            t_next = t + 1 if t < num_frames - 1 else 0
            
            f_curr = video_arr[t]
            f_next = video_arr[t_next]
            l_curr = label_arr[t]
            
            img_pair_np = np.stack([f_curr, f_next], axis=-1).astype(np.float32)
            label_np = l_curr.astype(np.uint8)
            
            if self.transforms:
                img_trans, label_trans = self.transforms(img_pair_np, label_np)
            else:
                img_trans = torch.from_numpy(img_pair_np).permute(2, 0, 1).float()
                label_trans = torch.from_numpy(label_np).long()
            
            frame_pairs.append(img_trans)
            labels.append(label_trans)

        return torch.stack(frame_pairs), torch.stack(labels)

    def __len__(self):
        return len(self.patient_list)

def ted_collate_fn(batch):
    batch_frames = []
    batch_labels = []
    for frames, labels in batch:
        batch_frames.append(frames)
        batch_labels.append(labels)
    return torch.cat(batch_frames, dim=0), torch.cat(batch_labels, dim=0)

def get_ted_transform(out_size=672):
    return Compose([
        Resize(out_size),
        ToTensor()
    ])