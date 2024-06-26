import torch
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from utils.io import is_available_to_store
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.RFRNet import RFRNet, VGG16FeatureExtractor
import os
import time
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
class RFRNetModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0
        self.writer = None
        self.totensor = transforms.ToTensor()

    def initialize_model(self, path=None, train=True, model_save_path = None, gpu_ids=[0]):
        self.G = RFRNet()
        if torch.cuda.device_count()>1:
            print(f"Using {len(gpu_ids)} gpus in parallel.")
            self.G = nn.DataParallel(self.G)#, device_ids = gpu_ids)

        self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-5)

        #tensorboard log dir
        if train:
            self.writer = SummaryWriter(os.path.join("logs", os.path.basename(model_save_path)))

        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-5)
                print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:
            print('No trained model, from start')
            self.iter = 0

    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")
    def save_img(self, img, file_path):
        '''

        img_min = np.min(img)
        img_max = np.max(img)

        #min-max normalization
        out_img = (img-img_min)/(img_max-img_min)
        out_img = out_img*255
        '''
        img = img.cpu().numpy()
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = img *255.
        cv2.imwrite(file_path, gray_img)


    def train(self, train_loader, save_path, store_capacity=10, finetune = False, iters=800000):
    #    writer = SummaryWriter(log_dir="log_info")
        count = 0
        self.G.train()
        if finetune:
            for param in self.G.parameters():
                param.requires_grad = False
            for param in self.G.module.fc.parameters():  # assuming `fc` is the part you want to finetune
                param.requires_grad = True
            self.optm_G = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=5e-5)
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()

        # while self.iter<iters:
        while True:
            for items in train_loader:
                gt_images, masks = self.__cuda__(*items)
               # print(masks)
               # masks = masks//255.0
                #print(masked_images.shape)
                #gt_images = gt_images.unsqueeze(dim=1)
                #masks = masks.unsqueeze(dim=1)

                masked_images = gt_images * masks
                #masked_images = masked_images.unsqueeze(dim=1)


                masked_image, fake_B, comp_B = self.forward(masked_images, masks, gt_images)

                self.update_parameters()
                self.iter += 1

                if self.iter % 5000 == 0:  # pth 저장되는 단위
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, time_taken:%.2f" %(self.iter, self.l1_loss_val/50, int_time))
                    self.writer.add_scalar("Train/Loss", self.l1_loss_val, self.iter)
                    s_time = time.time()
                    self.l1_loss_val = 0.0
                #store checkpoint per 40000 iteration.
                if self.iter % 10000==0 :
                    file_path = '{:s}/gt_img_{:d}.png'.format(f'./model/training/{os.path.basename(save_path)}', self.iter)
                    save_image(gt_images, file_path)

                    file_path = '{:s}/img_{:d}.png'.format(f'./model/training/{os.path.basename(save_path)}', self.iter)
                    save_image(comp_B, file_path)

                    file_path = '{:s}/f_img_{:d}.png'.format(f'./model/training/{os.path.basename(save_path)}', self.iter)
                    save_image(fake_B, file_path)

                    file_path = '{:s}/masked_img_{:d}.png'.format(f'./model/training/{os.path.basename(save_path)}',self.iter)
                    save_image(masked_image, file_path)

                    file_path = '{:s}/mask_{:d}.png'.format(f'./model/training/{os.path.basename(save_path)}',self.iter)
                    save_image(masks, file_path)
                if self.iter % 100 == 0:
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))
                    if is_available_to_store(store_capacity):
                        save_ckpt('{:s}/g_{:d}.pth'.format(save_path, self.iter ), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
                    else:
                        exit()
        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            save_ckpt('{:s}/g_{:s}.pth'.format(save_path, "final"), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
    def test(self, test_loader, result_save_path):
        self.G.eval()
        #print("++++checkintpoint1++++")
        for para in self.G.parameters():
            para.requires_grad = False
        count = 0
        #print("++++checkintpoint2++++")
        s_time = time.time()
        for items in test_loader:
            #print(test_loader)
            #print("++++checkintpoint3++++")
            si_time  = time.time()
            gt_images, masks= self.__cuda__(*items)

            #masks = masks//255.0
            masked_images = gt_images * masks
            #masked_images = gt_images

            #masks = torch.cat([masks]*3, dim = 1)
            result_degree_save_path = os.path.join(result_save_path, 'degree')

            folders = ['recon', 'gt', 'mask']
            # for fold in folders:
            #     temp = os.path.join(result_save_path, fold)
            #     if not os.path.isdir(temp):
            #         os.makedirs(temp)
            #     temp = os.path.join(result_degree_save_path, fold)
            #     if not os.path.isdir(temp):
            #         os.makedirs(temp)

            fake_B, mask = self.G(masked_images, masks)
            comp_B = fake_B * (1 - masks) + gt_images * masks
            if not os.path.exists('{:s}/'.format(result_save_path)):
                os.makedirs('{:s}/'.format(result_save_path))

            for k in range(fake_B.size(0)):
                count += 1
                #grid = make_grid(comp_B[k:k+1])
                # fake = fake_B[k:k+1]
                fake = comp_B[k:k+1]
                #print(fake)

                fake_degree = fake.squeeze()

                fake_degree = torch.sum(fake_degree, 0) / 3

                file_path = '{:s}/img_{:d}'.format(result_degree_save_path+'/recon', count)
                np.savetxt(file_path, np.array(fake_degree.cpu()),  delimiter=",")

                file_path = '{:s}/gt_{:d}'.format(result_degree_save_path+'/gt', count)
                np.savetxt(file_path,np.array(gt_images[0,1,:,:].cpu()),  delimiter=",")

                file_path = '{:s}/mask_{:d}'.format(result_degree_save_path+'/mask', count)
                np.savetxt(file_path,np.array(masks[0,1,:,:].cpu()),  delimiter=",")


                file_path = '{:s}/img_{:d}.png'.format(result_save_path+'/recon', count)
                save_image(fake, file_path)

                file_path = '{:s}/gt_images{:d}.png'.format(result_save_path+'/gt', count)
                save_image(gt_images, file_path)

                file_path = '{:s}/masked_img_{:d}.png'.format(result_save_path+'/mask', count)
                save_image(masked_images, file_path)

                #self.save_img(fake, file_path)
                #grid = make_grid(masked_images[k:k+1] +1 - masks[k:k+1] )
                #file_path = '{:s}/masked_img_{:d}.png'.format(result_save_path, count)
                #save_image(grid, file_path)
            ei_time  = time.time()
            i_time = ei_time-si_time
            print("img#"+str(count)+" : %.2f"%(i_time))
        e_time = time.time()
        int_time = e_time - s_time
        print("total time taken : %.2f"%(int_time))

    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
        return masked_image, self.fake_B, self.comp_B
    def update_parameters(self):
        self.update_G()
        self.update_D()

    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()

    def update_D(self):
        return

    def Gray2VGGInput(self, x, dtype):
        #normalized [0-1]

        x = torch.flatten(x, 1)
        #x = x.unsqueeze(dim=1)
        x_min, _ = torch.min(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        #print(x_min.shape)
        x = ((x-x_min) / (x_max-x_min)) * 255
        #print(x.shape)
        x = x.reshape(6, 256, 256)
        #print(x.shape)

        x_group =[]
        for i in range(6):
            x_img = x[i:i+1,:,:]
            x_img = torch.squeeze(x_img, 0)
            #print(x_img.shape)

            with torch.no_grad():
                x_img = Image.fromarray(np.uint8(x_img.cpu())).convert('RGB')
            x_tensor = self.totensor(x_img).cuda()
            x_tensor = torch.unsqueeze(x_tensor, 0)
            x_group.append(x_tensor)

        x_tensor = torch.cat(x_group, dim=0)

        #x_img = np.array(x_img)

        #x =  torch.cat([x]*3, dim = 1)
        #print(x.shape)
        return x_tensor

    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B

        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)

        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))

        loss_G = (  tv_loss * 0.1
                  + style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 1
                  + hole_loss * 6)

        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        return loss_G

    def l1_loss(self, f1, f2, mask = 1):
        return torch.mean(torch.abs(f1 - f2)*mask)

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value

    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value

    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)

