import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, input_dim, dprop):
        super(FC, self).__init__()
        self.input_dim = input_dim
        self.dprop = dprop

        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 1)

        # self.loss_fn = nn.NLLLoss()
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = lambda pred, target: torch.mean(torch.pow((target - pred), 2)) #L2 loss

    def forward(self, x):
        out = self.fc1(x)
        out = F.dropout(out, p=self.dprop, training=True, inplace=True)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        # out = self.log_softmax(out)
        return out