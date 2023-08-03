import torch.nn.functional as F
import torch.nn as nn
import torch

################################################################
#  Residual Block
################################################################
class ResidualBlock(nn.Module):
    """
    A residual block
    """

    def __init__(self, channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
                                   
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
        
        self.bn1 = nn.BatchNorm1d(num_features=channels)
        self.bn2 = nn.BatchNorm1d(num_features=channels)
        
    def forward(self, x):
        residual = x
        
        out = F.gelu(self.conv1(x))
        out = self.bn1(out)
        
        out = F.gelu(self.conv2(out))
        out = self.bn2(out)
        
        out = out + residual
        
        return out

##########################
# Small Resnet9
##########################
class ResNet9_small(nn.Module):
    """
    A Residual network.
    """
    def __init__(self,out_features,pool_size=5,kernel_size=11):
        super(ResNet9_small, self).__init__()
        
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        
        self.rb1 = ResidualBlock(channels=64, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.bn3 = nn.BatchNorm1d(num_features=96)
        
        self.conv4 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        
        self.rb2 = ResidualBlock(channels=128, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)

        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(in_features=128, out_features = out_features, bias=True)

    def forward(self, x):
        x = x[:,None,:]
        batch_size = len(x)
        
        
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.bn2(x)
        
        ##################
        # 1st residual
        ##################
        
        x = F.avg_pool1d(x,kernel_size=self.pool_size,stride=self.pool_size)
        x = self.rb1(x)
        
        x = self.conv3(x)
        x = F.gelu(x)
        x = self.bn3(x)
        
        x = F.avg_pool1d(x,kernel_size=self.pool_size,stride=self.pool_size)
        
        x = self.conv4(x)
        x = F.gelu(x)
        x = self.bn4(x)
        
        ##################
        # 2nd residual
        ##################
        
        x = F.avg_pool1d(x,kernel_size=self.pool_size,stride=self.pool_size)
        x = self.rb2(x)
                
        x = self.gap(x)
        x = x.view(batch_size,-1)
        
        
        out = self.fc(x)
        
        return out

##########################
# Larger Resnet9
##########################
class ResNet9_large(nn.Module):
    """
    A Residual network.
    """
    def __init__(self,out_features,pool_size=5,kernel_size=11):
        super(ResNet9_large, self).__init__()
        
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        
        self.rb1 = ResidualBlock(channels=128, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        
        self.rb2 = ResidualBlock(channels=512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)

        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(in_features=512, out_features = out_features, bias=True)

    def forward(self, x):
        x = x[:,None,:]
        batch_size = len(x)
        
        
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.bn2(x)
        
        ##################
        # 1st residual
        ##################
        
        x = F.avg_pool1d(x,kernel_size=self.pool_size,stride=self.pool_size)
        x = self.rb1(x)
        
        x = self.conv3(x)
        x = F.gelu(x)
        x = self.bn3(x)
        
        x = F.avg_pool1d(x,kernel_size=self.pool_size,stride=self.pool_size)
        
        x = self.conv4(x)
        x = F.gelu(x)
        x = self.bn4(x)
        
        ##################
        # 2nd residual
        ##################
        
        x = F.avg_pool1d(x,kernel_size=self.pool_size,stride=self.pool_size)
        x = self.rb2(x)
                
        x = self.gap(x)
        x = x.view(batch_size,-1)
        
        
        out = self.fc(x)
        
        return out

