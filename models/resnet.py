import torch
from torch import nn

class building_block(nn.Module):
    
    def __init__(self,in_channels, out_channels, stride = 1):
        
        super(building_block, self).__init__()
        
        
        self.conv_cell = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,  kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.skip_connection = None
        
        if (stride!=1) or (in_channels == out_channels):
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        
        precomputed = self.conv_cell(x)
        residual = self.skip_connection(x) if self.skip_connection else x
        
        return torch.relu(precomputed+residual)
    

class resnet(nn.Module):
    
    def __init__(self, conv_lens,channels, classes = 10):
        
        super(resnet, self).__init__()
        
        self.pre_conv = nn.Conv2d(3, channels[0], kernel_size = 3, padding = 1)
        self.pre_norm = nn.BatchNorm2d(channels[0])
        
        self.res_layer1 = self._build_resudual_seq(channels[0], channels[0], 1, conv_lens[0])
        self.res_layer2 = self._build_resudual_seq(channels[0], channels[1], 2, conv_lens[1])
        self.res_layer3 = self._build_resudual_seq(channels[1], channels[2], 2, conv_lens[2])
        
        self.downsample = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], classes)
        
    
    @staticmethod
    def _build_resudual_seq(in_channels, out_channels, stride, num_blocks):
        
        layer_list = []
        
        layer_list.append(building_block(in_channels, out_channels, stride))
        
        for _ in range(num_blocks-1):
            layer_list.append(building_block(out_channels, out_channels, 1))
            
        return nn.Sequential(*layer_list)
    
    def forward(self, x):
        
        x = torch.relu(self.pre_norm(self.pre_conv(x)))
        
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)

        x = self.fc(self.downsample(x).squeeze(2).squeeze(2))
        return x
        
        
def resnet20():
    return resnet([3,3,3], [16,32,64])