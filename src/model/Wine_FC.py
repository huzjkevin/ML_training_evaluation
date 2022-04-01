'''
 * User: Hojun Lim
 * Date: 2020-04-25
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Wine_FC(nn.Module):
    def __init__(self, input_dim, prop):
        super(Wine_FC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50,1)
        self.dropout_prop = prop
        self.dropout = nn.Dropout(p = prop)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    #TODO add dropout
