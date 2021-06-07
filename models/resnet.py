'''ResNet & VGG in PyTorch.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
import numpy as np
from torch.distributions import Normal
import sys
sys.path.append(".")
from config import BATCH
from .query_models import GraphConvolution
import math

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class ViTResNetfm(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dim = 128, num_tokens = 16, mlp_dim = 256, heads = 4, depth = 2, emb_dropout = 0.1, dropout= 0.1):
        super(ViTResNetfm, self).__init__()
        
        self.L = num_tokens
        self.cT = dim

        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)   
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.apply(_weights_init)
        
        

        self.transformer = Transformer3(BATCH, 1, 1, 128, dropout)
        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(512, num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
    
        
    def forward(self, img, mask = None):
        x = F.relu(self.bn1(self.conv1(img)))
        x = self.layer1(x)
        x = self.layer2(x)  
        x = self.layer3(x) 
        x = self.layer4(x) 
        x = F.avg_pool2d(x, 4)

        x_in = rearrange(x, 'b c h w -> (c h w) b ')
        
        # Transformer
        x_in = self.transformer(x_in, mask) #main game
        feat = rearrange(x_in, '(c h w)  b -> b c h w', 
                         b=BATCH, c=x.size(1), h=1, w=1)

        feat = feat.reshape(feat.size(0), -1)
        x = self.nn1(feat)
        
        
        return x, feat, feat

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.dropout(F.relu(self.bn1(self.conv1(x))), p=0.3, training=True)
        out = F.dropout(self.bn2(self.conv2(out)), p=0.3, training=True)
        out +=  self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
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
    def __init__(self, block, num_blocks, num_classes=10, avg_pool=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.avg_pool = avg_pool

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear = nn.Linear(512, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, self.avg_pool)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)
        out = self.linear(outf)
        return out, outf, [out1, out2, out3, out4]

class ResNetfm(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetfm, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)
        out = self.linear(outf)
        return out, outf, [out1, out2, out3, out4]


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        a = self.fn(x, **kwargs)
        return a + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block3(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.ReLU() #GELU()
        self.nn2 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)

        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.nn2(x)
        
        return x

class Attention3(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        self.to_qk = nn.Linear(dim, dim * 2, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        self.nn1 = nn.Linear(1024, 512)
        torch.nn.init.xavier_uniform_(self.to_qk.weight)
        torch.nn.init.zeros_(self.to_qk.bias)    
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        n, _, h = *x.shape, self.heads
        # qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        qk =  self.to_qk(x)
        # q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions
        q, k = rearrange(qk, 'n (qk h d) -> qk h n d', qk = 2, h = h)
        dots = torch.einsum('hid,hjd->hij', q, k) #* self.scale
        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = rearrange(attn, 'h n d -> n (h d) ') #concat heads into one matrix, ready for next encoder block  
        # out =  self.nn1(out)
        
        out = torch.einsum('ij,jk->ik', out, x) #* self.scale
        # out = self.do1(out)
        return out


class Transformer3(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention3(dim, heads = heads, dropout = dropout)),
                Residual(MLP_Block3(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            att_out = attention(x, mask = mask) # go to attention
            x = mlp(att_out) #go to MLP_Block
        return x


class GCNT(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCNT, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)        
        return x


def aff_to_adj(x, adj_identity, y=None):
    adj = torch.mm(x, torch.transpose(x, 0, 1))
    adj +=  -1.0 * adj_identity 
    adj_diag = torch.sum(adj, dim=0) #rowise sum
    adj = torch.mm(adj, torch.diag(1/adj_diag))
    adj = adj + adj_identity

    return adj

class ViTResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, avg_pool=4, dim = 512, num_tokens = 512, mlp_dim = 128, heads = 1, depth = 1, emb_dropout = 0.1, dropout= 0.1):
        super(ViTResNet, self).__init__()
        
        self.L = num_tokens
        self.cT = dim
        self.avg_pool = avg_pool


        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)   
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.apply(_weights_init)
        
        self.transformer = Transformer3(BATCH, depth, heads, mlp_dim, dropout)
        
        self.nn2 = nn.Linear(dim, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
    
        
    def forward(self, img, mask = None):
        x = F.relu(self.bn1(self.conv1(img)))
        x = self.layer1(x)
        x = self.layer2(x)  
        x = self.layer3(x) 
        x = self.layer4(x) 
        x = F.avg_pool2d(x, 4)

        x_in = rearrange(x, 'b c h w -> (c h w) b ')

        
        # Transformer
        x_in = self.transformer(x_in, mask) #main game
        feat = rearrange(x_in, '(c h w)  b -> b c h w', 
                         b=BATCH, c=x.size(1), h=1, w=1)


        feat = feat.reshape(feat.size(0), -1)
        y = self.nn2(feat)
        
        return y, feat, x.reshape(x.size(0), -1)


def ResNet18(num_classes = 10, avg_pool = 4):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, avg_pool)

def ViTResNet18(num_classes = 10, avg_pool = 4):
    # return ViTResNet(BasicBlock, [3,3,3], num_classes)
    return ViTResNet(BasicBlock, [2,2,2,2], num_classes, avg_pool)

def ResNet18fm(num_classes = 10):
    return ResNetfm(BasicBlock, [2,2,2,2], num_classes)

def ViTResNet18fm(num_classes = 10):
    # return ViTResNet(BasicBlock, [3,3,3], num_classes)
    return ViTResNetfm(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes = 10, avg_pool = 4):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, avg_pool)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, classes):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        feat = []
        y = x
        for i, model in enumerate(self.features):
            y = model(y)
            if i in {3,5,15,20}:
            # if i in {3,5,10,15}:
                feat.append(y)#(y.view(y.size(0),-1))

        x = self.features(x)
        out4 = x.view(x.size(0), -1)
        x = self.classifier(out4)
        return x, out4, [feat[0], feat[1], feat[2], feat[3]]


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VTVGG(nn.Module):
    '''
    VTVGG model 
    '''
    def __init__(self, features, classes):
        super(VTVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        self.transformer = Transformer3(BATCH, 1, 1, 128, 0.1)

    def forward(self, x, mask = None):
        feat = []
        y = x
        for i, model in enumerate(self.features):
            y = model(y)
            if i in {3,5,10,15}:
                
                feat.append(y)#(y.view(y.size(0),-1))

        x = self.features(x)
        x = rearrange(x, 'b c h w -> (c h w) b') # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP

        x = self.transformer(x, mask) #main game
        x = rearrange(x, '(c h w) b -> b c h w ', c = 512, h = 1, w = 1)

        out4 = x.view(x.size(0), -1)
        x = self.classifier(out4)
        return x, out4, [feat[0], feat[1], feat[2], feat[3]]


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(classes):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), classes)

def vgg16(classes):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), classes)

def VTvgg16(classes):
    """VGG 16-layer model (configuration "D")"""
    return VTVGG(make_layers(cfg['D']), classes)



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        feat = out.view(out.size(0), -1)
        out = self.linear(feat)

        return out, feat, feat

class VTWide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(VTWide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.L = 2560

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.transformer = Transformer3(BATCH, 1, 1, 128, 0.1)
        # self.token_wA = nn.Parameter(torch.empty(self.L, BATCH),requires_grad = True).cuda() #Tokenization parameters
        # torch.nn.init.xavier_uniform_(self.token_wA)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out_in = rearrange(out, 'b c h w -> (c h w) b') # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP

        out = self.transformer(out_in, mask) #main game
        out = rearrange(out, '(c h w) b -> b c h w ', c = 640, h = 1, w = 1)
        feat = out.view(out.size(0), -1)
        out = self.linear(feat)

        return out, feat, feat

def Wide_ResNet28(num_classes = 10):
    return Wide_ResNet(28, 10, 0.3, num_classes)

def VTWide_ResNet28(num_classes = 10):
    return VTWide_ResNet(28, 10, 0.3, num_classes)
