import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def _tensor_to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    def compute_dice(self, pred_mask, true_mask):
        pred = self._tensor_to_numpy(pred_mask)
        true = self._tensor_to_numpy(true_mask)
        
        dice_scores = []
        for c in range(self.num_classes):
            p = (pred == c)
            t = (true == c)
            
            intersection = np.sum(p & t)
            union = np.sum(p) + np.sum(t)
            
            if union == 0:
                score = 1.0 
            else:
                score = (2.0 * intersection) / (union + 1e-8)
            
            dice_scores.append(score)
            
        return dice_scores

    def compute_hausdorff_distance(self, pred_mask, true_mask):
        pred = self._tensor_to_numpy(pred_mask)
        true = self._tensor_to_numpy(true_mask)
        
        hd_scores = []
        for c in range(self.num_classes):
            pred_points = np.argwhere(pred == c)
            true_points = np.argwhere(true == c)
            
            if len(pred_points) > 0 and len(true_points) > 0:
                d_forward = directed_hausdorff(pred_points, true_points)[0]
                d_backward = directed_hausdorff(true_points, pred_points)[0]
                max_d = max(d_forward, d_backward)
                hd_scores.append(max_d)
            elif len(true_points) == 0 and len(pred_points) == 0:
                hd_scores.append(0.0)
            else:
                hd_scores.append(np.nan) 
                
        return hd_scores