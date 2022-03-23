import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models


# In[ ]:


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels=512, out_channels=512):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)        
