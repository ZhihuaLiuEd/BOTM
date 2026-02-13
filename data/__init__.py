from .camus import CAMUSDataset, get_camus_transform
from .ted import TEDDataset, ted_collate_fn, get_ted_transform

def get_dataset(cfg, split='train'):
    """
    split: 'train', 'val', or 'test'
    """
    dataset_name = cfg['dataset']['name']
    image_size = cfg['dataset']['image_size']
    
    if split == 'test':
        root = cfg['test']['data_root']
        is_train = False
    else:
        root = cfg['train']['data_root']
        is_train = True

    use_aug = (split == 'train')
    

    if dataset_name == 'CAMUS':
        view = cfg['dataset'].get('view', '2CH')
        transforms = get_camus_transform(is_train=use_aug, out_size=image_size)
        
        val_ratio = cfg['train'].get('val_split_ratio', 0.1) if cfg['train'].get('val_split') else 0.0
        return CAMUSDataset(root=root, view=view, transforms=transforms, split=split, val_ratio=val_ratio)
        
    elif dataset_name == 'TED':
        transforms = get_ted_transform(out_size=image_size)
        val_ratio = cfg['train'].get('val_split_ratio', 0.1) if cfg['train'].get('val_split') else 0.0
        return TEDDataset(root=root, transforms=transforms, split=split, val_ratio=val_ratio)
        
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")