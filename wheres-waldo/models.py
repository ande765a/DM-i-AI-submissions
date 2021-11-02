import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
  def __init__(self, in_channels):
    super(BaselineCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5, padding=2)
    self.conv2 = nn.Conv2d(10 + in_channels, 20, kernel_size=5, padding=2)
    self.conv3 = nn.Conv2d(10 + in_channels + 20 + in_channels, 1, kernel_size=1)

  def forward(self, x):
      c1 = self.conv1(x)
      c1 = F.relu(c1)
      c1 = torch.concat((c1, x), dim=1)

      c2 = self.conv2(c1)
      c2 = F.relu(c2)
      c2 = torch.concat((c2, c1, x), dim=1)

      c3 = self.conv3(c2)
      return c3