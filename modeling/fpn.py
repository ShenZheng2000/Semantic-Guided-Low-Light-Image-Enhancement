from modeling.resnet import resnet50
import torch.nn as nn
import torch.nn.functional as F
import torch

class fpn_module(nn.Module):
    def __init__(self, numClass):
        super(fpn_module, self).__init__()
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Classify layers
        self.classify = nn.Conv2d(128*4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.upsample(p5, size=(H, W), mode='bilinear')
        p4 = F.upsample(p4, size=(H, W), mode='bilinear')
        p3 = F.upsample(p3, size=(H, W), mode='bilinear')
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p5 = self.smooth1_2(self.smooth1_1(p5))
        p4 = self.smooth2_2(self.smooth2_1(p4))
        p3 = self.smooth3_2(self.smooth3_1(p3))
        p2 = self.smooth4_2(self.smooth4_1(p2))
        # Classify
        output = self.classify(self._concatenate(p5, p4, p3, p2))

        return output


class fpn(nn.Module):
    def __init__(self, numClass):
        super(fpn, self).__init__()
        # Res net
        self.resnet = resnet50(True)

        # fpn module
        self.fpn = fpn_module(numClass)

        # init fpn
        for m in self.fpn.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Top-down
        c2, c3, c4, c5 = self.resnet.forward(x)
        return self.fpn.forward(c2, c3, c4, c5)

class FocalLoss(nn.Module):

    # def __init__(self, device, gamma=0, eps=1e-7, size_average=True):
    def __init__(self, gamma=0, eps=1e-7, size_average=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.reduce = reduce
        # self.device = device

    def forward(self, input, target):
        # y = one_hot(target, input.size(1), self.device)
        y = one_hot(target, input.size(1))
        probs = F.softmax(input, dim=1)
        probs = (probs * y).sum(1)  # dimension ???
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.reduce:
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
        else:
            loss = batch_loss
        return loss

def one_hot(index, classes):
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]

    # mask = torch.Tensor(size).fill_(0).to(device)
    if torch.cuda.is_available():
        mask = torch.Tensor(size).fill_(0).cuda()
    else:
        mask = torch.Tensor(size).fill_(0)
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)

def get_NoGT_target(inputs):
    sfmx_inputs = F.log_softmax(inputs, dim=1)
    target = torch.argmax(sfmx_inputs, dim=1)
    return target

def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_CUBIC)
    return new_target
