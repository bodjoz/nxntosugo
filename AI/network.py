import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        # Circular padding handles the Torus topology seamlessly
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class TorusGoNet(nn.Module):
    def __init__(self, size=4, channels=64, num_res_blocks=3):
        super(TorusGoNet, self).__init__()
        self.size = size
        self.action_size = size * size + 1
        
        # Initial convolutional block
        self.conv = nn.Conv2d(2, channels, kernel_size=3, padding=1, padding_mode='circular')
        self.bn = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.pi_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.pi_bn = nn.BatchNorm2d(2)
        self.pi_fc = nn.Linear(2 * size * size, self.action_size)
        
        # Value Head
        self.v_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * size * size, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """x input tensor shape: [Batch, 2, size, size]"""
        # Common layers
        out = F.relu(self.bn(self.conv(x)))
        for block in self.res_blocks:
            out = block(out)
            
        # Policy Head
        pi = F.relu(self.pi_bn(self.pi_conv(out)))
        pi = pi.view(pi.size(0), -1)
        pi = self.pi_fc(pi)
        
        # Value Head
        v = F.relu(self.v_bn(self.v_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        
        return pi, v
