import argparse
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import get_dataset, ted_collate_fn
from models.botm_segformer import get_botm_model
from utils.loss import CE_Dice_Loss
from utils.metrics import Evaluator

def get_args():
    parser = argparse.ArgumentParser(description='Train BOTM for Echo Segmentation')
    parser.add_argument('--config', type=str, default='configs/camus_2ch.yaml', help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=0) 
    return parser.parse_args()

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        torch.save(state, os.path.join(save_dir, 'model_best.pth'))

def validate(model, val_loader, evaluator, device, dataset_name):
    model.eval()
    all_dices = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if dataset_name == 'CAMUS':
                es_img, es_label, ed_img, ed_label = [x.to(device) for x in batch]
                inputs = (es_img, ed_img)
                targets = (es_label, ed_label)
            elif dataset_name == 'TED':
                frames, labels = [x.to(device) for x in batch]
                inputs = (frames[:, 0].unsqueeze(1), frames[:, 1].unsqueeze(1))
                targets = (labels,)

            preds = model(*inputs)
            
            batch_dices = []
            num_eval_pairs = len(targets)
            
            for i in range(num_eval_pairs):
                pred_logits = preds[i]
                true_mask = targets[i]
                pred_mask = torch.argmax(pred_logits, dim=1)
                
                for b in range(pred_mask.shape[0]):
                    scores = evaluator.compute_dice(pred_mask[b], true_mask[b])
                    batch_dices.append(scores)
            
            all_dices.extend(batch_dices)

    all_dices = np.array(all_dices) 
    class_dices = np.mean(all_dices, axis=0)

    if len(class_dices) > 1:
        avg_dice = np.mean(class_dices[1:])
    else:
        avg_dice = np.mean(class_dices)
        
    return avg_dice, class_dices

def main():
    args = get_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = cfg.get('experiment_name', 'default_exp')
    save_dir = os.path.join(cfg['train']['save_dir'], experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'config_backup.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    print(f"Start Experiment: {experiment_name}")
    print(f"Save Directory: {save_dir}")

    dataset_name = cfg['dataset']['name']
    collate_fn = ted_collate_fn if dataset_name == 'TED' else None
    
    train_set = get_dataset(cfg, split='train')
    
    use_val = cfg['train'].get('val_split', True) 
    val_set = get_dataset(cfg, split='val') if use_val else None

    train_loader = DataLoader(
        train_set, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['train']['num_workers'], 
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if val_set:
        val_loader = DataLoader(
            val_set, 
            batch_size=1,
            shuffle=False, 
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn
        )
    else:
        val_loader = None
        print("Warning: No validation set configured.")

    model = get_botm_model(cfg['model']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg['train']['lr'], momentum=0.9)
    criterion = CE_Dice_Loss(num_classes=cfg['model']['num_classes']).to(device)
    evaluator = Evaluator(num_classes=cfg['model']['num_classes'])

    start_epoch = 0
    best_score = 0.0
    if cfg['train'].get('resume'):
        ckpt_path = cfg['train']['resume']
        if os.path.isfile(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint.get('best_score', 0.0)
        else:
            print(f"Checkpoint not found at {ckpt_path}, starting from scratch.")

    epochs = cfg['train']['epochs']
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            loss = 0.0

            if dataset_name == 'CAMUS':
                es_img, es_label, ed_img, ed_label = [x.to(device) for x in batch]
                pred_es, pred_ed = model(es_img, ed_img)
                loss_es = criterion(pred_es, es_label)
                loss_ed = criterion(pred_ed, ed_label)
                loss = loss_es + loss_ed
            elif dataset_name == 'TED':
                frames, labels = [x.to(device) for x in batch]
                img_t = frames[:, 0].unsqueeze(1)
                img_t1 = frames[:, 1].unsqueeze(1)
                pred_t, pred_t1 = model(img_t, img_t1)
                loss = criterion(pred_t, labels)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

        is_best = False
        if val_loader and (epoch + 1) % cfg['train'].get('val_interval', 1) == 0:
            val_score, class_scores = validate(model, val_loader, evaluator, device, dataset_name)
            print(f"Validation Mean Dice: {val_score:.4f}")
            print(f"    Class Scores: {[f'{s:.4f}' for s in class_scores]}")
            
            if val_score > best_score:
                best_score = val_score
                is_best = True
                print("New Best Model!")

        save_interval = cfg['train'].get('save_interval', 10)
        if (epoch + 1) % save_interval == 0 or is_best:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_score': best_score,
                'config': cfg
            }, is_best, save_dir, filename=f'ckpt_epoch_{epoch+1}.pth')

    print("Training Finished.")

if __name__ == '__main__':
    main()