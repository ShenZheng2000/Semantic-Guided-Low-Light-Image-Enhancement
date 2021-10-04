import os
import Myloss
import dataloader
from modeling import model
import torch.optim
from modeling.fpn import *
from option import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU only
device = get_device()

class Trainer():
    def __init__(self):
        self.scale_factor = args.scale_factor
        self.net = model.enhance_net_nopool(self.scale_factor, conv_type=args.conv_type).to(device)
        self.seg = fpn(args.num_of_SegClass).to(device)
        self.seg_criterion = FocalLoss(gamma=2).to(device)
        self.train_dataset = dataloader.lowlight_loader(args.lowlight_images_path)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True)
        self.L_color = Myloss.L_color()
        self.L_spa = Myloss.L_spa8(patch_size=args.patch_size)
        self.L_exp = Myloss.L_exp(16)
        self.L_TV = Myloss.L_TV()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.num_epochs = args.num_epochs
        self.E = args.exp_level
        self.grad_clip_norm = args.grad_clip_norm
        self.display_iter = args.display_iter
        self.snapshot_iter = args.snapshot_iter
        self.snapshots_folder = args.snapshots_folder

        if args.load_pretrain == True:
            self.net.load_state_dict(torch.load(args.pretrain_dir, map_location=device))
            print("weight is OK")


    def get_seg_loss(self, enhanced_image):
        # segment the enhanced image
        seg_input = enhanced_image.to(device)
        seg_output = self.seg(seg_input).to(device)

        # build seg output
        target = (get_NoGT_target(seg_output)).data.to(device)

        # calculate seg. loss
        seg_loss = self.seg_criterion(seg_output, target)

        return seg_loss


    def get_loss(self, A, enhanced_image, img_lowlight, E):
        Loss_TV = 1600 * self.L_TV(A)
        loss_spa = torch.mean(self.L_spa(enhanced_image, img_lowlight))
        loss_col = 5 * torch.mean(self.L_color(enhanced_image))
        loss_exp = 10 * torch.mean(self.L_exp(enhanced_image, E))
        loss_seg = self.get_seg_loss(enhanced_image)

        loss = Loss_TV + loss_spa + loss_col + loss_exp + 0.1 * loss_seg

        return loss


    def train(self):
        self.net.train()

        for epoch in range(self.num_epochs):

            for iteration, img_lowlight in enumerate(self.train_loader):

                img_lowlight = img_lowlight.to(device)
                enhanced_image, A = self.net(img_lowlight)
                loss = self.get_loss(A, enhanced_image, img_lowlight, self.E)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.net.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                if ((iteration + 1) % self.display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())
                if ((iteration + 1) % self.snapshot_iter) == 0:
                    torch.save(self.net.state_dict(), self.snapshots_folder + "Epoch" + str(epoch) + '.pth')



if __name__ == "__main__":
    t = Trainer()
    t.train()









