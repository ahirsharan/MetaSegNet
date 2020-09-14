import torch.nn as nn
import torch
import math

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,dilations=[1,1,1],down=True):
        super(BasicBlock, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        
        stride=1
        if(down):
            stride=2
                   
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilations[0])
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        
        self.conv2 = conv3x3(planes, planes,dilation=dilations[1])
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1,inplace=True)        
        
        
        self.conv3 = conv3x3(planes, planes,dilation=dilations[2])
        self.bn3 = norm_layer(planes)      
        self.relu3 = nn.LeakyReLU(negative_slope=0.1,inplace=True)             
        
        self.down=down
        if(self.down):
            self.downsample = conv1x1(inplanes,planes,stride)
           
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if(self.down):
            identity=self.downsample(x)
            
        out+=identity
        out = self.relu3(out)

        return out
    
class resnet9(nn.Module):
    def __init__(self, num_channels,out_channels):
        super(resnet9, self).__init__()    
        
        self.num_channels=num_channels
        self.out_channels=out_channels
        self.Conv=nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1=nn.BatchNorm2d(num_features=64)
        self.relu1=nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool0=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.resblock1=BasicBlock(inplanes=64, planes=64, down=False)
        
        self.resblock2=BasicBlock(inplanes=64, planes=128)
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        
        self.resblock3=BasicBlock(inplanes=128, planes=256,dilations=[1,2,4])
    
        #Local Feature
        self.resblock4=BasicBlock(inplanes=256,planes=512,dilations=[8,16,32])
        
        #Global Context        
        self.pool2=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.resblock5=BasicBlock(inplanes=256,planes=512)
        self.pool4=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.pool5=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        
        self.bn2=nn.BatchNorm2d(num_features=1024)
        self.relu2=nn.ReLU(inplace=True)

        #Adjusting Dimensions
        self.blockadd=nn.Conv2d(1024, 960, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3=nn.BatchNorm2d(num_features=960)
        self.relu3=nn.ReLU(inplace=True)  
        
    def forward(self,x):
        
        out=self.Conv(x)
        out=self.bn1(out)
        out=self.relu1(out)
        out=self.pool0(out)
        
        out=self.resblock1(out)
        
        out=self.resblock2(out)
        out=self.pool1(out)
        
        out=self.resblock3(out)
        
        out1=out
        out1=self.resblock4(out1)
        
        out2=out
        out2=self.pool2(out2)
        out2=self.pool3(out2)
        out2=self.resblock5(out2)
        out2=self.pool4(out2)
        out2=self.pool5(out2)
        
        #Concatenate
        out=torch.cat((out1,out2),1)
        out=self.bn2(out)
        out=self.relu2(out)
        
        #For adjusting dimensions
        out=self.blockadd(out)
        out=self.bn3(out)
        out=self.relu3(out)

        print("Output: ")
        print(out.shape)
        #Reshape to segmentation map
        out1=out.reshape(out.shape[0],-1,self.out_channels)
        out=out.reshape((-1,self.out_channels))
        return out,out1
