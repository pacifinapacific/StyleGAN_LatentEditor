import numpy as np 
import matplotlib.pyplot as plt 
from stylegan_layers import  G_mapping,G_synthesis
from read_image import image_reader
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.utils import save_image
from perceptual_model import VGG16_for_Perceptual
import torch.optim as optim

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
     parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
     parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
     parser.add_argument('--resolution',default=1024,type=int)
     parser.add_argument('--src_im1',default="source_image/joker.png")
     parser.add_argument('--src_im2',default="source_image/0.png")
     parser.add_argument('--mask',default="source_image/Blur_mask.png")
     parser.add_argument('--weight_file',default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt",type=str)
     parser.add_argument('--iteration',default=1500,type=int)



     args=parser.parse_args()

     g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(resolution=args.resolution))    
    ]))

     g_all.load_state_dict(torch.load(args.weight_file, map_location=device))
     g_all.eval()
     g_all.to(device)
     g_mapping,g_synthesis=g_all[0],g_all[1]


     img_0=image_reader(args.src_im1) #(1,3,1024,1024) -1~1
     img_0=img_0.to(device)

     img_1=image_reader(args.src_im2)
     img_1=img_1.to(device) #(1,3,1024,1024)

     blur_mask0=image_reader(args.mask).to(device)
     blur_mask0=blur_mask0[:,0,:,:].unsqueeze(0)
     blur_mask1=blur_mask0.clone()
     blur_mask1=1-blur_mask1

     MSE_Loss=nn.MSELoss(reduction="mean")
     upsample2d=torch.nn.Upsample(scale_factor=0.5, mode='bilinear')

     img_p0=img_0.clone() #resize for perceptual net
     img_p0=upsample2d(img_p0)
     img_p0=upsample2d(img_p0) #(1,3,256,256)

     img_p1=img_1.clone()
     img_p1=upsample2d(img_p1)
     img_p1=upsample2d(img_p1) #(1,3,256,256)




     perceptual_net=VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device) #conv1_1,conv1_2,conv2_2,conv3_3
     dlatent=torch.zeros((1,18,512),requires_grad=True,device=device)
     optimizer=optim.Adam({dlatent},lr=0.01,betas=(0.9,0.999),eps=1e-8)


     print("Start")
     loss_list=[]
     for i in range(args.iteration):
          optimizer.zero_grad()
          synth_img=g_synthesis(dlatent)
          synth_img = (synth_img + 1.0) / 2.0
          loss_wl0=caluclate_loss(synth_img,img_0,perceptual_net,img_p0,blur_mask0,MSE_Loss,upsample2d)
          loss_wl1=caluclate_loss(synth_img,img_1,perceptual_net,img_p1,blur_mask1,MSE_Loss,upsample2d)
          loss=loss_wl0+loss_wl1
          loss.backward()

          optimizer.step()

          loss_np=loss.detach().cpu().numpy()
          loss_0=loss_wl0.detach().cpu().numpy()
          loss_1=loss_wl1.detach().cpu().numpy()

          loss_list.append(loss_np)
          if i%10==0:
               print("iter{}: loss -- {},  loss0 --{},  loss1 --{}".format(i,loss_np,loss_0,loss_1))
               save_image(synth_img.clamp(0,1),"save_image/crossover/{}.png".format(i))
               np.save("latent_W/crossover.npy",dlatent.detach().cpu().numpy())
          






          



def caluclate_loss(synth_img,img,perceptual_net,img_p,blur_mask,MSE_Loss,upsample2d): #W_l
     #calculate MSE Loss
     mse_loss=MSE_Loss(synth_img*blur_mask.expand(1,3,1024,1024),img*blur_mask.expand(1,3,1024,1024)) # (lamda_mse/N)*||G(w)-I||^2
     #calculate Perceptual Loss
     real_0,real_1,real_2,real_3=perceptual_net(img_p)
     synth_p=upsample2d(synth_img) #(1,3,256,256)
     synth_p=upsample2d(synth_p)
     synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)
     #print(synth_0.size(),synth_1.size(),synth_2.size(),synth_3.size())

     perceptual_loss=0
     blur_mask=upsample2d(blur_mask)
     blur_mask=upsample2d(blur_mask) #(256,256)

     perceptual_loss+=MSE_Loss(synth_0*blur_mask.expand(1,64,256,256),real_0*blur_mask.expand(1,64,256,256))
     perceptual_loss+=MSE_Loss(synth_1*blur_mask.expand(1,64,256,256),real_1*blur_mask.expand(1,64,256,256))
     blur_mask=upsample2d(blur_mask) 
     blur_mask=upsample2d(blur_mask) #(64,64)
     perceptual_loss+=MSE_Loss(synth_2*blur_mask.expand(1,256,64,64),real_2*blur_mask.expand(1,256,64,64))
     blur_mask=upsample2d(blur_mask) #(64,64)
     perceptual_loss+=MSE_Loss(synth_3*blur_mask.expand(1,512,32,32),real_3*blur_mask.expand(1,512,32,32))



     return mse_loss+perceptual_loss




     



     



     

    

if __name__ == "__main__":
    main()



