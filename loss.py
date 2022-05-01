import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://www.compuphase.com/cmetric.htm
# expects values from 0-1 as rgb
def ColourDistMean(prd, tgt):
    rmean = (prd[:,0] * 255 - tgt[:,0] * 255) / 2
    rdiff = prd[:,0] - tgt[:,0]
    gdiff = prd[:,1] - tgt[:,1]
    bdiff = prd[:,2] - tgt[:,2]
    loss = torch.sqrt((2 + (rmean / 256)) * rdiff**2 + 4 * gdiff**2 + (2 + (255 - rmean) / 256) * bdiff**2)
    return torch.mean(loss)
def ColourDistSum(prd, tgt):
    rmean = (prd[:,0] * 255 - tgt[:,0] * 255) / 2
    rdiff = prd[:,0] - tgt[:,0]
    gdiff = prd[:,1] - tgt[:,1]
    bdiff = prd[:,2] - tgt[:,2]
    loss = torch.sqrt((2 + (rmean / 256)) * rdiff**2 + 4 * gdiff**2 + (2 + (255 - rmean) / 256) * bdiff**2)
    return torch.sum(loss)
    
    
def ColourWeightedMSELoss(prd, tgt):
    rloss = F.mse_loss(prd[:,0], tgt[:,0])
    gloss = F.mse_loss(prd[:,1], tgt[:,1])
    bloss = F.mse_loss(prd[:,2], tgt[:,2])
    
    return 0.35 * rloss + 0.4 * gloss + 0.25 * bloss