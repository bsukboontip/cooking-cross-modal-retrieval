import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_loss(x1, x2):
    """
    The cosine loss is calculated as the sum of the cosine distance between the anchor and positive image, and the anchor and negative image.
    """

    return 1 - x1.mm(x2)

def multi_label_loss(logits, labels):
    """
    Formulate the soft-cross entropy as a multi-label loss function.
    """
    return F.binary_cross_entropy_with_logits(logits, labels)

class TripletLoss(nn.Module):
    """Triplet loss class
    Parameters
    ----------
    margin : float
        Ranking loss margin
    metric : string
        Distance metric (either euclidean or cosine)
    """

    def __init__(self, margin=0.3):

        super(TripletLoss, self).__init__()
        self.distance_function = cosine_loss
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, im, s):
        # compute image-sentence score matrix
        # batch_size x batch_size
        scores_i2r = self.distance_function(F.normalize(im, dim=-1), F.normalize(s, dim=-1))
        scores_r2i = scores_i2r.t()

        pos = torch.eye(im.size(0))
        neg = 1 - pos

        pos = (pos == 1).to(im.device)
        neg = (neg == 1).to(im.device)

        # positive similarities
        # batch_size x 1
        d1 = scores_i2r.diag().view(im.size(0), 1)
        d2 = d1.t()

        y = torch.ones(scores_i2r.size(0)).to(im.device)


        # image anchor - recipe positive bs x bs
        d1 = d1.expand_as(scores_i2r)
        # recipe anchor - image positive
        d2 = d2.expand_as(scores_i2r) #bs x bs

        y = y.expand_as(scores_i2r)

        # compare every diagonal score to scores in its column
        # recipe retrieval
        # batch_size x batch_size (each anchor is compared to all elements in the batch)
        cost_im = self.ranking_loss(scores_i2r, d1, y)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_s = self.ranking_loss(scores_i2r, d2, y)

        # clear diagonals
        cost_s = cost_s.masked_fill_(pos, 0)
        cost_im = cost_im.masked_fill_(pos, 0)

        return (cost_s + cost_im).mean()