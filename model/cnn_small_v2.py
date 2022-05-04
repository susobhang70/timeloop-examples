import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch2timeloop

class CNNSmallV2(nn.Module):
    def __init__(self, num_classes=1):
        super(CNNSmallV2, self).__init__()

        self.num_classes = num_classes

        # defining batchnorm input                                                                                               
        # self.batchnorm1 = nn.BatchNorm2d(1)

        # defining Convolutional layers                                                                                           
        conv1_C = 128
        # conv2_C = 512
        conv3_C = 256
        # conv4_C = 2048
        # conv5_C = hparams['out_channels'][4]
        conv_W = 18
        pad = int(np.floor(conv_W / 2) - 1)
        stride_W = 2
        pool_W = 3
        self.conv1 = nn.Conv2d(1, conv1_C, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.bn1 = nn.BatchNorm2d(conv1_C)
        # self.conv2 = nn.Conv2d(conv1_C, conv2_C, kernel_size=(1,conv_W), stride=(1,1), padding=(0,pad))
        # self.bn2 = nn.BatchNorm2d(conv2_C)
        self.conv3 = nn.Conv2d(conv1_C, conv3_C, kernel_size=(1,conv_W), stride=(1,1), padding=(0,pad))
        self.bn3 = nn.BatchNorm2d(conv3_C)
        # self.conv4 = nn.Conv2d(conv3_C, conv4_C, kernel_size=(1,conv_W), stride=(1,1), padding=(0,pad))
        # self.bn4 = nn.BatchNorm2d(conv4_C)
        # self.conv5 = nn.Conv2d(conv4_C, conv5_C, kernel_size=(1,conv_W), stride=(1,1), padding=(0,pad))

        # Defining pooling                                                                                                        
        pad = int(np.floor(pool_W / 2))
        self.pool = nn.MaxPool2d(kernel_size=(1,pool_W), stride=(1,stride_W),padding=(0,pad))
        
        # Defining pooling                                                                                                        
        pad = int(np.floor(pool_W / 2))
        self.pool = nn.MaxPool2d(kernel_size=(1,pool_W), stride=(1,stride_W),padding=(0,pad))

        # Defining output layer                                                                                                   
        input_W = 2 ** 11
        num_layers = 3
        conv_C = 256
        
        if num_layers == 0:
            num_layers = 1
        # hard coded, figure it out manually
        map_W = 511                                                                                  
        embedding_dim = int(conv_C)
        self.fc1 = nn.Linear(embedding_dim, num_classes)
        self.fc2 = nn.Sigmoid()
        
        # Defining global average pooling                                                                                         
        # self.poolMean = nn.AvgPool2d(kernel_size=(1,map_W), stride=(1,map_W), padding=(0,0))

        # Defining global max pooling                                                                                             
        self.poolMax = nn.MaxPool2d(kernel_size=(1,map_W), stride=(1,map_W), padding=(0,0))
    
    def forward(self, x):
        x = x.transpose(1, 2)
        # targets = batch['targets']
        # if x.dim() == 3:
        x = x.unsqueeze(1)
        
        # x = self.batchnorm1(x)
        x = self.bn1(F.relu(self.conv1(x)))
        # x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.bn3(F.relu(self.conv3(x)))
        # x = self.pool(x)
        # print(x.shape)
        # x = self.bn4(F.relu(self.conv4(x)))
        x = self.poolMax(x)
        x = x.squeeze(2)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        

net = CNNSmallV2()
input_shape = (1024, 40) #~5sec

# Define the number of batches that will be used for the inference
batch_size = 1

# Define the directory names where the timeloop workload yaml files will be stored.
# The yaml files will be stored in ./workloads/alexnet/ in this example.
top_dir = 'workloads'
sub_dir = 'cnn_small_v2'

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