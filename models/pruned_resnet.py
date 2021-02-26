import torch
from torch import nn

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        
class pruned_conv(nn.Module):
    
    def __init__(self,in_channels, centroids, cluster_mapping, kernel_size = 3, stride = 1, padding = 1, bias=False):
        
        super(pruned_conv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, centroids.shape[0], kernel_size = kernel_size,
                              stride = stride, padding = padding, bias = bias)
        
        self.conv.weight = nn.Parameter(centroids)
        self.cluster_mapping = cluster_mapping
        
        
    def forward(self, x):
        
        x = self.conv(x)
        x = [x[:,i,:,:] for i in self.cluster_mapping]
        x = torch.stack(x, dim=1)
        
        return x

class pruned_building_block(nn.Module):
    
    def __init__(self,convs_params, stride = 1):
        
        super(pruned_building_block, self).__init__()
        
        
        self.conv_cell = nn.Sequential(
            pruned_conv(*convs_params[0], kernel_size = 3, stride = stride, padding = 1, bias=False),
            nn.BatchNorm2d(convs_params[0][2].shape[0]),
            nn.ReLU(),
            pruned_conv(*convs_params[1],  kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(convs_params[1][2].shape[0])
        )
        
        self.skip_connection = None
        
        if (stride!=1):# or (in_channels != out_channels):
            self.skip_connection = lambda x: torch.nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, convs_params[-1][-1].shape[0]//4, convs_params[-1][-1].shape[0]//4), "constant", 0)
        
    def forward(self, x):
        
        precomputed = self.conv_cell(x)
        residual = self.skip_connection(x) if self.skip_connection else x
        
        return torch.relu(precomputed+residual)
    

class pruned_resnet(nn.Module):
    
    def __init__(self, conv_lens,parameters ,classes = 10):
        
        super(pruned_resnet, self).__init__()
        
#         assert len(conv_lens)==len(channels)
        
        self.pre_conv = pruned_conv(*parameters[0], kernel_size = 3, padding = 1, bias=False)
        self.pre_norm = nn.BatchNorm2d(parameters[0][2].shape[0])
        
        self.res_layer1 = self._build_resudual_seq(parameters[1], 1, conv_lens[0])
        self.res_layer2 = self._build_resudual_seq(parameters[2], 2, conv_lens[1])
        self.res_layer3 = self._build_resudual_seq(parameters[3], 2, conv_lens[2])
        
        self.downsample = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(parameters[-1][-1][-1][-1].shape[0], classes)
        
        
    
    @staticmethod
    def _build_resudual_seq(block_params, stride, num_blocks):
        
        layer_list = []
        
        layer_list.append(pruned_building_block(block_params[0], stride))
        
        for i in range(1,num_blocks):
            layer_list.append(pruned_building_block(block_params[i], 1))
            
        return nn.Sequential(*layer_list)
    
    def forward(self, x):
        
        x = torch.relu(self.pre_norm(self.pre_conv(x)))
        
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)

        x = self.fc(self.downsample(x).squeeze(2).squeeze(2))
        return x
        
        
def pruned_resnet20(parameters):
    return pruned_resnet([3,3,3], parameters)