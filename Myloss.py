import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
        #self.eps = 1e-6
        self.eps = 0

    def forward(self, x ):

        #b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2) + self.eps,0.5)


        return k

			
class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).to(device).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        #weight_diff =torch.max(torch.FloatTensor([1]).to(device) + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).to(device),torch.FloatTensor([0]).to(device)),torch.FloatTensor([0.5]).to(device))
        #E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(device)) ,enhance_pool-org_pool)


        # Original output
        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        # Enhanced output
        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        # Difference
        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

# New spa loss
class L_spa8(nn.Module):
    def __init__(self, patch_size):
        super(L_spa8, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # Build conv kernels
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_upleft = torch.FloatTensor( [[-1,0,0],[0,1,0],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_upright = torch.FloatTensor( [[0,0,-1],[0,1,0],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_loleft = torch.FloatTensor( [[0,0,0],[0,1,0],[-1,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_loright = torch.FloatTensor( [[0,0,0],[0,1,0],[0,0,-1]]).to(device).unsqueeze(0).unsqueeze(0)

        # convert to parameters
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.weight_upleft = nn.Parameter(data=kernel_upleft, requires_grad=False)
        self.weight_upright = nn.Parameter(data=kernel_upright, requires_grad=False)
        self.weight_loleft = nn.Parameter(data=kernel_loleft, requires_grad=False)
        self.weight_loright = nn.Parameter(data=kernel_loright, requires_grad=False)

        # pooling layer
        self.pool = nn.AvgPool2d(patch_size) # default is 4

    def forward(self, org , enhance ):
        #b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        #weight_diff =torch.max(torch.FloatTensor([1]).to(device) + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).to(device),torch.FloatTensor([0]).to(device)),torch.FloatTensor([0.5]).to(device))
        #E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(device)) ,enhance_pool-org_pool)


        # Original output
        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)
        D_org_upleft = F.conv2d(org_pool , self.weight_upleft , padding=1)
        D_org_upright = F.conv2d(org_pool , self.weight_upright, padding=1)
        D_org_loleft = F.conv2d(org_pool , self.weight_loleft, padding=1)
        D_org_loright = F.conv2d(org_pool , self.weight_loright, padding=1)


        # Enhanced output
        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)
        D_enhance_upleft = F.conv2d(enhance_pool, self.weight_upleft, padding=1)
        D_enhance_upright = F.conv2d(enhance_pool, self.weight_upright, padding=1)
        D_enhance_loleft = F.conv2d(enhance_pool, self.weight_loleft, padding=1)
        D_enhance_loright = F.conv2d(enhance_pool, self.weight_loright, padding=1)

        # Difference
        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        D_upleft = torch.pow(D_org_upleft - D_enhance_upleft,2)
        D_upright = torch.pow(D_org_upright - D_enhance_upright,2)
        D_loleft = torch.pow(D_org_loleft - D_enhance_loleft,2)
        D_loright = torch.pow(D_org_loright - D_enhance_loright,2)

        # Total difference
        E = (D_left + D_right + D_up +D_down) + 0.5 * (D_upleft + D_upright + D_loleft + D_loright)

        # E = 25*(D_left + D_right + D_up +D_down)

        return E

# l2 exposure loss
class L_exp(nn.Module):

    def __init__(self,patch_size):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val
    def forward(self, x, mean_val ):

        #b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        meanTensor = torch.FloatTensor([mean_val] ).to(device)

        d = torch.mean(torch.pow(mean- meanTensor,2))
        return d

# Smooth l1 loss
class L1_exp(nn.Module):

    def __init__(self, patch_size):
        super(L1_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val

    def forward(self, x, mean_val):
        # b,c,h,w = x.shape
        crit = torch.nn.SmoothL1Loss()

        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        meanTensor = torch.FloatTensor([mean_val]).to(device)

        #d = torch.mean(torch.pow(mean - meanTensor, 2))
        d = torch.mean(crit(mean, meanTensor))
        return d

class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)

        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
