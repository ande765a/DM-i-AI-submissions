import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import  resnet18



from receptive_field import receptive_field


class TransferModel2(nn.Module):
  def __init__(self):
    super(TransferModel2, self).__init__()
    self.resnet18 = resnet18(pretrained=True)

    self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=2)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=2)

  def forward(self, x):
    x = self.resnet18.conv1(x)
    x = self.resnet18.bn1(x)
    x = self.resnet18.relu(x)
    x = self.resnet18.maxpool(x)
    x = self.resnet18.layer1(x)
    #x = self.resnet18.layer2(x)
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    
    return x

class TransferModel(nn.Module):
  def __init__(self):
    super(TransferModel, self).__init__()
    self.resnet18 = resnet18(pretrained=True)

    # for param in self.resnet18.parameters():
    #   param.requires_grad = False

    self.conv1 = nn.Conv2d(128, 32, kernel_size=4)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(32, 1, kernel_size=1)

  def forward(self, x):
    x = self.resnet18.conv1(x)
    x = self.resnet18.bn1(x)
    x = self.resnet18.relu(x)
    x = self.resnet18.maxpool(x)
    x = self.resnet18.layer1(x)
    x = self.resnet18.layer2(x)
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    
    return x

class SimpleCNN(nn.Module):
  def __init__(self, in_channels):
    super(SimpleCNN, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, 16, kernel_size=3),
      nn.Conv2d(16, 16, kernel_size=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=2),
      nn.Conv2d(32, 32, kernel_size=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=5, stride=2),
      nn.Conv2d(64, 64, kernel_size=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=5, stride=2),
      nn.Conv2d(128, 1, kernel_size=1, padding=0),
    )
    
  def forward(self, x):
    return self.conv(x)

if __name__ == "__main__":
  model = TransferModel()
  out = model(torch.zeros(1, 3, 32, 32))
  print(out.shape)
  out = model(torch.zeros(1, 3, 1500, 1500))
  print(out.shape)
  #receptive_field(model.to(torch.device("cuda:0")), (3, 1500, 1500))


class ResBlock2d(nn.Module):
  def __init__(self, channels, **kwargs):
    super(ResBlock2d, self).__init__()
    
    self.cnn1 = nn.Conv2d(
        in_channels=channels, 
        out_channels=channels,
        **kwargs
    )
    
    self.bn1 = nn.BatchNorm2d(num_features=channels)
    
    self.cnn2 = nn.Conv2d(
        in_channels=channels, 
        out_channels=channels,
        **kwargs
    )
    
    self.bn2 = nn.BatchNorm2d(num_features=channels)
    
  def forward(self, X):
    out = self.cnn1(X)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.cnn2(out)
    out = self.bn2(out)
    out = X + out
    out = F.relu(out)
    return out



class BaselineCNN(nn.Module):
  def __init__(self, in_channels):
      super(BaselineCNN, self).__init__()
      self.part1 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        ResBlock2d(32, kernel_size=5, padding=2),
      )

      self.part2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=9, padding=4),
        ResBlock2d(64, kernel_size=5, padding=2),
        nn.Conv2d(64, 1, kernel_size=1)
      )
    

  def forward(self, x):
      out = self.part1(x)
      return self.part2(out)

class ResConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(ResConv2d, self).__init__()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    self.bn = nn.BatchNorm2d(num_features=out_channels)
    self.resblock = ResBlock2d(channels=out_channels, **kwargs)

  def forward(self, X):
    out = self.conv(X)
    out = self.bn(out)
    out = F.relu(out)
    out = self.resblock(out)
    return out

class DoubleConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(DoubleConv2d, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
    self.bn1 = nn.BatchNorm2d(num_features=out_channels)

    self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, **kwargs)
    self.bn2 = nn.BatchNorm2d(num_features=out_channels)

  def forward(self, X):
    out = self.conv1(X)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = F.relu(out)

    return out

class Down(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(Down, self).__init__()
    self.mp = nn.MaxPool2d(kernel_size=2)
    self.conv = DoubleConv2d(in_channels=in_channels, out_channels=out_channels, **kwargs) 

  def forward(self, X):
    out = self.mp(X)
    out = self.conv(out)
    return out

class Up(nn.Module):
  def __init__(self, in_channels, skip_channels, out_channels, **kwargs):
    super(Up, self).__init__()
    self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    self.conv = DoubleConv2d(in_channels=in_channels + skip_channels, out_channels=out_channels, **kwargs)

  def forward(self, X, skip):
    out = self.upsample(X)
    out = torch.cat((out, skip), dim=1)
    out = self.conv(out)
    return out

class UNet(nn.Module):
  def __init__(self, in_channels):
    super(UNet, self).__init__()
    self.conv_in = DoubleConv2d(in_channels=in_channels, out_channels=64, kernel_size=1)
  
    # Downsample
    self.down1 = Down(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.down2 = Down(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    #self.down3 = Down(in_channels=256, out_channels=512, kernel_size=3, padding=1)
    
    # Upsample
    #self.up2 = Up(in_channels=512, skip_channels=256, out_channels=256, kernel_size=3, padding=1)
    self.up3 = Up(in_channels=256, skip_channels=128, out_channels=128, kernel_size=3, padding=1)
    self.up4 = Up(in_channels=128, skip_channels=64, out_channels=64, kernel_size=3, padding=1)

    # Output for mask and has_subtitle
    self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

  def forward(self, X):
    skip1 = self.conv_in(X)
    skip2 = self.down1(skip1)
    latent = self.down2(skip2)
    #latent = self.down3(skip3)

    # Upsample
    #out = self.up2(latent, skip3)
    out = self.up3(latent, skip2)
    out = self.up4(out, skip1)
    
    # Output
    mask_logits = self.conv_out(out)
    
    return mask_logits