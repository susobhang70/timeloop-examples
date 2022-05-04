import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch2timeloop
import torchvision

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class EffNetOri(nn.Module):
    def __init__(self, label_dim=1, pretrain=False, model_id=0, audioset_pretrain=False):
        super().__init__()
        b = int(model_id)
        print('now train a effnet-b{:d} model'.format(b))
        if b == 7:
            self.model = torchvision.models.efficientnet_b7(pretrained=pretrain)
        elif b == 6:
            self.model = torchvision.models.efficientnet_b6(pretrained=pretrain)
        elif b == 5:
            self.model = torchvision.models.efficientnet_b5(pretrained=pretrain)
        elif b == 4:
            self.model = torchvision.models.efficientnet_b4(pretrained=pretrain)
        elif b == 3:
            self.model = torchvision.models.efficientnet_b3(pretrained=pretrain)
        elif b == 2:
            self.model = torchvision.models.efficientnet_b2(pretrained=pretrain)
        elif b == 1:
            self.model = torchvision.models.efficientnet_b1(pretrained=pretrain)
        elif b == 0:
            self.model = torchvision.models.efficientnet_b0(pretrained=pretrain)
        new_proj = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # print('conv1 get from pretrained model.')
        new_proj.weight = torch.nn.Parameter(torch.sum(self.model.features[0][0].weight, dim=1).unsqueeze(1))
        new_proj.bias = self.model.features[0][0].bias
        self.model.features[0][0] = new_proj
        # self.model = create_feature_extractor(self.model, {'features.8': 'mout'})
        self.model.avgpool = Identity()
        self.model.classifier = Identity()
        # print(self.model)
        self.feat_dim, self.freq_dim = self.get_dim()
        # print(self.feat_dim)
        self.attention = MeanPooling(self.feat_dim, label_dim)

    def get_dim(self):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = torch.zeros(10, 1000, 40)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        # print(x.shape)
        x = self.model.features(x)
        # print(x.shape[1], x.shape[2])
        return int(x.shape[1]), int(x.shape[2])

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.model.features(x)
        out = torch.sigmoid(self.attention(x))
        return out

class MeanPooling(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layernorm = nn.LayerNorm(n_in)
        self.linear = nn.Linear(n_in, n_out)
        # print('use mean pooling')

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        x = torch.mean(x, dim=[2, 3])
        x = self.linear(x)
        return x
        
        

net = EffNetOri()
input_shape = (1024, 40) #~5sec

# Define the number of batches that will be used for the inference
batch_size = 1

# Define the directory names where the timeloop workload yaml files will be stored.
# The yaml files will be stored in ./workloads/alexnet/ in this example.
top_dir = 'workloads'
sub_dir = 'cnn_effnet'

# By default, nn.Conv2d modules will be automatically converted, but nn.Linear modules will be ignored.
# If you want to convert nn.Linear, set the option to be true.
# The converter will change the description of nn.Linear into Convolution-like layer.
# (e.g., in_channel=in_features, out_channel=out_features, input_height=1, input_width=1, filter size = 1x1, stride = 1x1, padding = 0x0)
# If you want to ignore nn.Linear layers, set this option to be false. 
convert_fc = True

# Finally, in case there exists a layer that is only used during the training phase, define an identifier for a such layer. 
# For example, in torchvision.models.inception_v3, auxiliary classification layers are not used during the inference (e.g., InceptionAux).
# In this case, include a string that can serve as an identifier for such layers (e.g., 'Aux') in exception_module_names.
# But for the above alexnet, there is no necessity to define this. 
exception_module_names = []

# Now, convert!
if __name__ == "__main__":
    pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)

# def count_parameters(model):
#     mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
#     mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
#     mem = mem_params + mem_bufs
#     return mem
#     # return sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(count_parameters(net))