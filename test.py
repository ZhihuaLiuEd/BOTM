import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import get_dataset, ted_collate_fn
from models.botm_segformer import get_botm_model
from utils.metrics import Evaluator

def get_args():
    parser = argparse.ArgumentParser(description='Test BOTM')
    parser.add_argument('--config', type=str, default='configs/camus_2ch.yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model_best.pth')
    return parser.parse_args()

def main():
    args = get_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_name = cfg['dataset']['name']
    collate_fn = ted_collate_fn if dataset_name == 'TED' else None
    test_set = get_dataset(cfg, split='test') 
    
    test_loader = DataLoader(
        test_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"Loading model from {args.ckpt}")
    model = get_botm_model(cfg['model']).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    evaluator = Evaluator(num_classes=cfg['model']['num_classes'])
    
    all_dices = []
    all_hds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if dataset_name == 'CAMUS':
                es_img, es_label, ed_img, ed_label = [x.to(device) for x in batch]
                inputs = (es_img, ed_img)
                targets = (es_label, ed_label)
            elif dataset_name == 'TED':
                frames, labels = [x.to(device) for x in batch]
                inputs = (frames[:, 0].unsqueeze(1), frames[:, 1].unsqueeze(1))
                targets = (labels,)

            preds = model(*inputs)
            
            num_eval_pairs = len(targets)
            
            for i in range(num_eval_pairs):
                pred_logits = preds[i]
                true_mask = targets[i]
                pred_mask = torch.argmax(pred_logits, dim=1)
                
                for b in range(pred_mask.shape[0]):
                    dices = evaluator.compute_dice(pred_mask[b], true_mask[b])
                    all_dices.append(dices)
                    
                    hds = evaluator.compute_hausdorff_distance(pred_mask[b], true_mask[b])
                    all_hds.append(hds)

    all_dices = np.array(all_dices)
    all_hds = np.array(all_hds)

    print("\n====== Test Results ======")
    class_names = [f"Class {i}" for i in range(cfg['model']['num_classes'])]
    
    print("Dice Scores:")
    mean_dice_per_class = np.mean(all_dices, axis=0)
    for name, score in zip(class_names, mean_dice_per_class):
        print(f"  {name}: {score:.4f}")
    print(f"  Average (w/o BG): {np.mean(mean_dice_per_class[1:]):.4f}")

    print("\nHausdorff Distances:")
    mean_hd_per_class = np.nanmean(all_hds, axis=0)
    for name, score in zip(class_names, mean_hd_per_class):
        print(f"  {name}: {score:.4f}")
    print(f"  Average (w/o BG): {np.nanmean(mean_hd_per_class[1:]):.4f}")

if __name__ == '__main__':
    main()