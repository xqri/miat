'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out1))
        out += self.shortcut(x)
        out = F.relu(out)
        return out, out1

class KMeansBatchNorm(nn.Module):
    def __init__(self, k=3, num_features=64, dimension=128, batchsize=32, momentum=0.1):
        super(KmeansNorm, self).__init__()
        self.k = k
        self.c = torch.randn(self.k, dimension).cuda()
        self.batchsize = batchsize
        self.momentum = momentum
        self.bn = [nn.BatchNorm2d(num_features) for i in range(self.k)]
        
        
    def forward(self, x):
        m = x.view(self.batchsize, -1)
        mc = torch.mm(m, torch.t(self.c))
        m2 = torch.sum(m ** 2, axis=1)
        c2 = torch.sum(c ** 2, axis=1)
        m2 = m2.repeat((self.k,1))
        c2 = c2.repeat((self.batchsize,1))
        diff = torch.abs(torch.t(m2) + c2 - 2*mc)
        ind = np.argsort(diff, axis=1) 
        out = torch.zeros_like(x).cuda()
        for i in range(self.k):
            mask = (ind == i)
            x_sub = x[mask]
            if x_sub!= []:
                x_sub = self.bn[i](x_sub)
                self.c[i,:] = self.momentum * torch.mean(m[mask], dim=0) + (1.0 - self.momentum) * self.c[i,:]
                out[mask] = x_sub
                
        return out
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv_lay1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay1_1 = nn.BatchNorm2d(64)
        self.conv_lay1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay1_2 = nn.BatchNorm2d(64)
        self.shortcut_lay1 = nn.Sequential()

        self.conv_lay2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay2_1 = nn.BatchNorm2d(64)
        self.conv_lay2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay2_2 = nn.BatchNorm2d(64)
        self.shortcut_lay2 = nn.Sequential()

        self.conv_lay3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_lay3_1 = nn.BatchNorm2d(128)
        self.conv_lay3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay3_2 = nn.BatchNorm2d(128)
        self.shortcut_lay3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.conv_lay4_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay4_1 = nn.BatchNorm2d(128)
        self.conv_lay4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay4_2 = nn.BatchNorm2d(128)
        self.shortcut_lay4 = nn.Sequential()

        self.conv_lay5_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_lay5_1 = nn.BatchNorm2d(256)
        self.conv_lay5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay5_2 = nn.BatchNorm2d(256)
        self.shortcut_lay5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.conv_lay6_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay6_1 = nn.BatchNorm2d(256)
        self.conv_lay6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay6_2 = nn.BatchNorm2d(256)
        self.shortcut_lay6 = nn.Sequential()

        self.conv_lay7_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_lay7_1 = nn.BatchNorm2d(512)
        self.conv_lay7_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay7_2 = nn.BatchNorm2d(512)
        self.shortcut_lay7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

            
        self.conv_lay8_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay8_1 = nn.BatchNorm2d(512)
        self.conv_lay8_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay8_2 = nn.BatchNorm2d(512)
        self.shortcut_lay8 = nn.Sequential()

        self.conv_lay8_1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay8_1_2 = nn.BatchNorm2d(512)
        self.conv_lay8_2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_lay8_2_2 = nn.BatchNorm2d(512)
        self.shortcut_lay8_2 = nn.Sequential()

        # self.layer1s = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2s = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3s = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4s = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear_2 = nn.Linear(512*block.expansion, num_classes)

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride).cuda())
    #         self.in_planes = planes * block.expansion
    #     return layers #nn.Sequential(*layers)
    '''
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        #out = self.layer3(out)
        return out
    '''
    def forward(self, x, branch=0):

        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = F.relu(self.bn_lay1_1(self.conv_lay1_1(out1)))
        out = self.bn_lay1_2(self.conv_lay1_2(out2))
        out += self.shortcut_lay1(out1)
        out3 = F.relu(out)
        
        out4 = F.relu(self.bn_lay2_1(self.conv_lay2_1(out3)))
        out = self.bn_lay2_2(self.conv_lay2_2(out4))
        out += self.shortcut_lay2(out3)
        out5 = F.relu(out)
        
        out6 = F.relu(self.bn_lay3_1(self.conv_lay3_1(out5)))
        out = self.bn_lay3_2(self.conv_lay3_2(out6))
        out += self.shortcut_lay3(out5)
        out7 = F.relu(out)
        
        out8 = F.relu(self.bn_lay4_1(self.conv_lay4_1(out7)))
        out = self.bn_lay4_2(self.conv_lay4_2(out8))
        out += self.shortcut_lay4(out7)
        out9 = F.relu(out)
        
        out10 = F.relu(self.bn_lay5_1(self.conv_lay5_1(out9)))
        out = self.bn_lay5_2(self.conv_lay5_2(out10))
        out += self.shortcut_lay5(out9)
        out11 = F.relu(out)
        
        out12 = F.relu(self.bn_lay6_1(self.conv_lay6_1(out11)))
        out = self.bn_lay6_2(self.conv_lay6_2(out12))
        out += self.shortcut_lay6(out11)
        out13 = F.relu(out)
        
        out14 = F.relu(self.bn_lay7_1(self.conv_lay7_1(out13)))
        out = self.bn_lay7_2(self.conv_lay7_2(out14))
        out += self.shortcut_lay7(out13)
        out15 = F.relu(out)
        
        if branch == 0:
            
            out16 = F.relu(self.bn_lay8_1(self.conv_lay8_1(out15)))
            out = self.bn_lay8_2(self.conv_lay8_2(out16))
            out += self.shortcut_lay8(out15)
            out17 = F.relu(out)
            
            out = F.avg_pool2d(out17, 4)
            out = out.view(out.size(0), -1)
            out18 = self.linear(out)
        
        else:
            
            out16 = F.relu(self.bn_lay8_1_2(self.conv_lay8_1_2(out15)))
            out = self.bn_lay8_2_2(self.conv_lay8_2_2(out16))
            out += self.shortcut_lay8_2(out15)
            out17 = F.relu(out)
    
            out = F.avg_pool2d(out17, 4)
            out = out.view(out.size(0), -1)
            out18 = self.linear_2(out)
            
        return out18
        
        # out = F.avg_pool2d(out17, 4)
        # out = out.view(out.size(0), -1)
        # out18 = self.linear(out)
        # return [out5, out7, out9, out11, out13, out18]#[out1, out3, out5, out7, out9, out11, out13, out15, out17, out18]#

    def forward1(self, out1):
        out2 = F.relu(self.bn_lay1_1(self.conv_lay1_1(out1)))
        out = self.bn_lay1_2(self.conv_lay1_2(out2))
        out += self.shortcut_lay1(out1)
        out3 = F.relu(out)

        out4 = F.relu(self.bn_lay2_1(self.conv_lay2_1(out3)))
        out = self.bn_lay2_2(self.conv_lay2_2(out4))
        out += self.shortcut_lay2(out3)
        out5 = F.relu(out)

        out6 = F.relu(self.bn_lay3_1(self.conv_lay3_1(out5)))
        out = self.bn_lay3_2(self.conv_lay3_2(out6))
        out += self.shortcut_lay3(out5)
        out7 = F.relu(out)

        out8 = F.relu(self.bn_lay4_1(self.conv_lay4_1(out7)))
        out = self.bn_lay4_2(self.conv_lay4_2(out8))
        out += self.shortcut_lay4(out7)
        out9 = F.relu(out)

        out10 = F.relu(self.bn_lay5_1(self.conv_lay5_1(out9)))
        out = self.bn_lay5_2(self.conv_lay5_2(out10))
        out += self.shortcut_lay5(out9)
        out11 = F.relu(out)

        out12 = F.relu(self.bn_lay6_1(self.conv_lay6_1(out11)))
        out = self.bn_lay6_2(self.conv_lay6_2(out12))
        out += self.shortcut_lay6(out11)
        out13 = F.relu(out)

        out14 = F.relu(self.bn_lay7_1(self.conv_lay7_1(out13)))
        out = self.bn_lay7_2(self.conv_lay7_2(out14))
        out += self.shortcut_lay7(out13)
        out15 = F.relu(out)

        out16 = F.relu(self.bn_lay8_1(self.conv_lay8_1(out15)))
        out = self.bn_lay8_2(self.conv_lay8_2(out16))
        out += self.shortcut_lay8(out15)
        out17 = F.relu(out)

        out = F.avg_pool2d(out17, 4)
        out = out.view(out.size(0), -1)
        out18 = self.linear(out)
        return [out5, out9, out13, out17, out18]

    def forward3(self, out3):
        out4 = F.relu(self.bn_lay2_1(self.conv_lay2_1(out3)))
        out = self.bn_lay2_2(self.conv_lay2_2(out4))
        out += self.shortcut_lay2(out3)
        out5 = F.relu(out)

        out6 = F.relu(self.bn_lay3_1(self.conv_lay3_1(out5)))
        out = self.bn_lay3_2(self.conv_lay3_2(out6))
        out += self.shortcut_lay3(out5)
        out7 = F.relu(out)

        out8 = F.relu(self.bn_lay4_1(self.conv_lay4_1(out7)))
        out = self.bn_lay4_2(self.conv_lay4_2(out8))
        out += self.shortcut_lay4(out7)
        out9 = F.relu(out)

        out10 = F.relu(self.bn_lay5_1(self.conv_lay5_1(out9)))
        out = self.bn_lay5_2(self.conv_lay5_2(out10))
        out += self.shortcut_lay5(out9)
        out11 = F.relu(out)

        out12 = F.relu(self.bn_lay6_1(self.conv_lay6_1(out11)))
        out = self.bn_lay6_2(self.conv_lay6_2(out12))
        out += self.shortcut_lay6(out11)
        out13 = F.relu(out)

        out14 = F.relu(self.bn_lay7_1(self.conv_lay7_1(out13)))
        out = self.bn_lay7_2(self.conv_lay7_2(out14))
        out += self.shortcut_lay7(out13)
        out15 = F.relu(out)

        out16 = F.relu(self.bn_lay8_1(self.conv_lay8_1(out15)))
        out = self.bn_lay8_2(self.conv_lay8_2(out16))
        out += self.shortcut_lay8(out15)
        out17 = F.relu(out)

        out = F.avg_pool2d(out17, 4)
        out = out.view(out.size(0), -1)
        out18 = self.linear(out)
        return out18#[out18]
        
    def forward5(self, out5):
        out6 = F.relu(self.bn_lay3_1(self.conv_lay3_1(out5)))
        out = self.bn_lay3_2(self.conv_lay3_2(out6))
        out += self.shortcut_lay3(out5)
        out7 = F.relu(out)

        out8 = F.relu(self.bn_lay4_1(self.conv_lay4_1(out7)))
        out = self.bn_lay4_2(self.conv_lay4_2(out8))
        out += self.shortcut_lay4(out7)
        out9 = F.relu(out)

        out10 = F.relu(self.bn_lay5_1(self.conv_lay5_1(out9)))
        out = self.bn_lay5_2(self.conv_lay5_2(out10))
        out += self.shortcut_lay5(out9)
        out11 = F.relu(out)

        out12 = F.relu(self.bn_lay6_1(self.conv_lay6_1(out11)))
        out = self.bn_lay6_2(self.conv_lay6_2(out12))
        out += self.shortcut_lay6(out11)
        out13 = F.relu(out)

        out14 = F.relu(self.bn_lay7_1(self.conv_lay7_1(out13)))
        out = self.bn_lay7_2(self.conv_lay7_2(out14))
        out += self.shortcut_lay7(out13)
        out15 = F.relu(out)

        out16 = F.relu(self.bn_lay8_1(self.conv_lay8_1(out15)))
        out = self.bn_lay8_2(self.conv_lay8_2(out16))
        out += self.shortcut_lay8(out15)
        out17 = F.relu(out)

        out = F.avg_pool2d(out17, 4)
        out = out.view(out.size(0), -1)
        out18 = self.linear(out)
        return out18#[out18]
        
    def forward7(self, out7):
        out8 = F.relu(self.bn_lay4_1(self.conv_lay4_1(out7)))
        out = self.bn_lay4_2(self.conv_lay4_2(out8))
        out += self.shortcut_lay4(out7)
        out9 = F.relu(out)

        out10 = F.relu(self.bn_lay5_1(self.conv_lay5_1(out9)))
        out = self.bn_lay5_2(self.conv_lay5_2(out10))
        out += self.shortcut_lay5(out9)
        out11 = F.relu(out)

        out12 = F.relu(self.bn_lay6_1(self.conv_lay6_1(out11)))
        out = self.bn_lay6_2(self.conv_lay6_2(out12))
        out += self.shortcut_lay6(out11)
        out13 = F.relu(out)

        out14 = F.relu(self.bn_lay7_1(self.conv_lay7_1(out13)))
        out = self.bn_lay7_2(self.conv_lay7_2(out14))
        out += self.shortcut_lay7(out13)
        out15 = F.relu(out)

        out16 = F.relu(self.bn_lay8_1(self.conv_lay8_1(out15)))
        out = self.bn_lay8_2(self.conv_lay8_2(out16))
        out += self.shortcut_lay8(out15)
        out17 = F.relu(out)

        out = F.avg_pool2d(out17, 4)
        out = out.view(out.size(0), -1)
        out18 = self.linear(out)
        return out18#[out18]

    def forward9(self, out9):
        out10 = F.relu(self.bn_lay5_1(self.conv_lay5_1(out9)))
        out = self.bn_lay5_2(self.conv_lay5_2(out10))
        out += self.shortcut_lay5(out9)
        out11 = F.relu(out)

        out12 = F.relu(self.bn_lay6_1(self.conv_lay6_1(out11)))
        out = self.bn_lay6_2(self.conv_lay6_2(out12))
        out += self.shortcut_lay6(out11)
        out13 = F.relu(out)

        out14 = F.relu(self.bn_lay7_1(self.conv_lay7_1(out13)))
        out = self.bn_lay7_2(self.conv_lay7_2(out14))
        out += self.shortcut_lay7(out13)
        out15 = F.relu(out)

        out16 = F.relu(self.bn_lay8_1(self.conv_lay8_1(out15)))
        out = self.bn_lay8_2(self.conv_lay8_2(out16))
        out += self.shortcut_lay8(out15)
        out17 = F.relu(out)

        out = F.avg_pool2d(out17, 4)
        out = out.view(out.size(0), -1)
        out18 = self.linear(out)
        return out18#[out18]

    def forward11(self, out11):
        out12 = F.relu(self.bn_lay6_1(self.conv_lay6_1(out11)))
        out = self.bn_lay6_2(self.conv_lay6_2(out12))
        out += self.shortcut_lay6(out11)
        out13 = F.relu(out)

        out14 = F.relu(self.bn_lay7_1(self.conv_lay7_1(out13)))
        out = self.bn_lay7_2(self.conv_lay7_2(out14))
        out += self.shortcut_lay7(out13)
        out15 = F.relu(out)

        out16 = F.relu(self.bn_lay8_1(self.conv_lay8_1(out15)))
        out = self.bn_lay8_2(self.conv_lay8_2(out16))
        out += self.shortcut_lay8(out15)
        out17 = F.relu(out)

        out = F.avg_pool2d(out17, 4)
        out = out.view(out.size(0), -1)
        out18 = self.linear(out)
        return out18#[out18]

    def forward13(self, out13):
        out14 = F.relu(self.bn_lay7_1(self.conv_lay7_1(out13)))
        out = self.bn_lay7_2(self.conv_lay7_2(out14))
        out += self.shortcut_lay7(out13)
        out15 = F.relu(out)

        out16 = F.relu(self.bn_lay8_1(self.conv_lay8_1(out15)))
        out = self.bn_lay8_2(self.conv_lay8_2(out16))
        out += self.shortcut_lay8(out15)
        out17 = F.relu(out)

        out = F.avg_pool2d(out17, 4)
        out = out.view(out.size(0), -1)
        out18 = self.linear(out)
        return out18#[out18]

    def forward15(self, out15):
        out16 = F.relu(self.bn_lay8_1(self.conv_lay8_1(out15)))
        out = self.bn_lay8_2(self.conv_lay8_2(out16))
        out += self.shortcut_lay8(out15)
        out17 = F.relu(out)

        out = F.avg_pool2d(out17, 4)
        out = out.view(out.size(0), -1)
        out18 = self.linear(out)
        return out18#[out18]

    def forward17(self, out17):
        out = F.avg_pool2d(out17, 4)
        out = out.view(out.size(0), -1)
        out18 = self.linear(out)
        return out18#[out18]

def ResNet18_2():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34_2():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50_2():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
