import torch
import torch.nn as nn
import torch.nn.functional as F

# Alternating Co-Attention obtained from the paper: https://arxiv.org/abs/1606.00061; 
# The code is taken from: https://github.com/jiasenlu/HieCoAttenVQA

class AlternatingCoAttention(nn.Module):
    """
    The Alternating Co-Attention module as in (Lu et al, 2017) paper Sec. 3.3.
    """
    def __init__(self, d=512, k=512, dropout=0.5):
        super().__init__()
        self.d = d
        self.k = k

        self.Wx1 = nn.Linear(d, k)
        self.whx1 = nn.Linear(k, 1)

        self.Wx2 = nn.Linear(d, k)
        self.Wg2 = nn.Linear(d, k)
        self.whx2 = nn.Linear(k, 1)

        self.Wx3 = nn.Linear(d, k)
        self.Wg3 = nn.Linear(d, k)
        self.whx3 = nn.Linear(k, 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, V):
        """
        Inputs:
            Q: question feature in a shape of BxTxd
            V: image feature in a shape of BxNxd
        Outputs:
            shat: attended question feature in a shape of Bxk
            vhat: attended image feature in a shape of Bxk
        """
        B = Q.shape[0]

        # 1st step
        H = torch.tanh(self.Wx1(Q))
        H = self.dropout(H)
        ax = F.softmax(self.whx1(H), dim=1)
        shat = torch.sum(Q * ax, dim=1, keepdim=True)

        # 2nd step
        H = torch.tanh(self.Wx2(V) + self.Wg2(shat))
        H = self.dropout(H)
        ax = F.softmax(self.whx2(H), dim=1)
        vhat = torch.sum(V * ax, dim=1, keepdim=True)

        # 3rd step
        H = torch.tanh(self.Wx3(Q) + self.Wg3(vhat))
        H = self.dropout(H)
        ax = F.softmax(self.whx3(H), dim=1)
        shat2 = torch.sum(Q * ax, dim=1, keepdim=True)

        return shat2.squeeze(), vhat.squeeze()