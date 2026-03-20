import torch
from torch import nn


class SoftJaccardLoss(nn.Module):
    def __init__(self, weight=1e-7):
        super(SoftJaccardLoss, self).__init__()
        self.weight = weight


    def forward(self, output, target):
        # Apply sigmoid to get probabilities (for binary segmentation)
        probs = torch.sigmoid(output)
        
        intersection = torch.sum(probs * target)
        union = torch.sum(probs) + torch.sum(target) - intersection
        
        jaccard_score = (intersection + self.weight) / (union + self.weight)
        
        return 1 - jaccard_score


class SoftJaccardBCELoss(nn.Module):
    def __init__(self, weight=1e-7):
        super(SoftJaccardBCELoss, self).__init__()
        self.weight = weight


    def forward(self, output, target):
        # Apply sigmoid to get probabilities (for binary segmentation)
        probs = torch.sigmoid(output)
        
        intersection = torch.sum(probs * target)
        union = torch.sum(probs) + torch.sum(target) - intersection
        
        jaccard_score = (intersection + self.weight) / (union + self.weight)
        
        BCECriterion = nn.BCEWithLogitsLoss()
        BCELoss = BCECriterion(output, target)

        return 1 - jaccard_score + 0.0001 * BCELoss


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=1e-7):
        super(SoftDiceLoss, self).__init__()
        self.weight = weight


    def forward(self, output, target):
        # Apply sigmoid to get probabilities (for binary segmentation)
        probs = torch.sigmoid(output)
 
        intersection = torch.sum(probs * target)
        union = torch.sum(probs) + torch.sum(target)

        dice_score = (2 * intersection + self.weight) / (union + self.weight)

        return 1 - dice_score


class SoftDiceBCELoss(nn.Module):
    def __init__(self, weight=1e-7):
        super(SoftDiceBCELoss, self).__init__()
        self.weight = weight


    def forward(self, output, target):
        # Apply sigmoid to get probabilities (for binary segmentation)
        probs = torch.sigmoid(output)
 
        intersection = torch.sum(probs * target)
        union = torch.sum(probs) + torch.sum(target)

        dice_score = (2 * intersection + self.weight) / (union + self.weight)

        BCECriterion = nn.BCEWithLogitsLoss()
        BCELoss = BCECriterion(output, target)

        return 1 - dice_score +  0.0001 * BCELoss

"""
class TverskyLoss(nn.Module):
    def __init__(self, alpha_t=0.5, beta_t=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        

    def forward(self, output, target, weight=1):
        probs = torch.sigmoid(output)

        TP = torch.sum()
        FN = torch.sum()
        FP = torch.sum()

        Tversky = (TP + weight) / (TP + self.alpha_t * FP + self.beta_t * FN + weight)
        print(self.alpha_t, self.beta_t)

        return 1 - Tversky
"""