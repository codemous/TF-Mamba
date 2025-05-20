import torch
from torch import nn
from torch.nn import functional as F


class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args['base']['alpha']
        self.Rec_Fn = ReconLoss(type=args['base']['rec_loss'])
        self.MSE_Fn = nn.MSELoss()

    def forward(self, out,label,mask):

        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels']) # task loss


        l_rec_low = self.Rec_Fn(out['rec_text'][0], out['complete_text'][0], mask) if out['rec_text'] is not None and out['complete_text'] is not None else 0


        loss = l_sp + self.alpha * l_rec_low

        return {'loss': loss, 'l_sp': l_sp, 'l_rec': l_rec_low}

class ReconLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.eps = 1e-6
        self.type = type
        if type == 'L1Loss':
            self.loss = nn.L1Loss(reduction='sum')
        elif type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2]).float()

        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + self.eps)

        return loss