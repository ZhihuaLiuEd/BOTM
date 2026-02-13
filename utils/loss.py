import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):

        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        if self.ignore_index is not None:
            mask = torch.ones(self.num_classes, device=logits.device, dtype=torch.bool)
            mask[self.ignore_index] = False
            dice_score = dice_score[mask]
            
        return 1.0 - torch.mean(dice_score)

class CE_Dice_Loss(nn.Module):
    def __init__(self, num_classes, weight_ce=0.5, weight_dice=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def forward(self, inputs, targets):
        targets = targets.long()
        
        loss_c = self.ce_loss(inputs, targets)
        loss_d = self.dice_loss(inputs, targets)
        
        return self.weight_ce * loss_c + self.weight_dice * loss_d

if __name__ == '__main__':
    B, C, H, W = 2, 4, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    
    criterion = CE_Dice_Loss(num_classes=C)
    loss = criterion(pred, target)
    print(f"Test Loss Value: {loss.item()}")