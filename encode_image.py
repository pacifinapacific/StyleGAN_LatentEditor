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
     parser.add_argument('--src_im',default="sample.png")
     parser.add_argument('--src_dir',default="source_image/")
     parser.add_argument('--weight_file',default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt",type=str)
     parser.add_argument('--iteration',default=1000,type=int)



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
     name=args.src_im.split(".")[0]
     img=image_reader(args.src_dir+args.src_im) #(1,3,1024,1024) -1~1
     img=img.to(device)

     MSE_Loss=nn.MSELoss(reduction="mean")

     img_p=img.clone() #Perceptual loss 用画像
     upsample2d=torch.nn.Upsample(scale_factor=256/args.resolution, mode='bilinear') #VGG入力のため(256,256)にリサイズ
     img_p=upsample2d(img_p)

     perceptual_net=VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device)
     dlatent=torch.zeros((1,18,512),requires_grad=True,device=device)
     optimizer=optim.Adam({dlatent},lr=0.01,betas=(0.9,0.999),eps=1e-8)

     print("Start")
     loss_list=[]
     for i in range(args.iteration):
          optimizer.zero_grad()
          synth_img=g_synthesis(dlatent)
          synth_img = (synth_img + 1.0) / 2.0
          mse_loss,perceptual_loss=caluclate_loss(synth_img,img,perceptual_net,img_p,MSE_Loss,upsample2d)
          loss=mse_loss+perceptual_loss
          loss.backward()

          optimizer.step()

          loss_np=loss.detach().cpu().numpy()
          loss_p=perceptual_loss.detach().cpu().numpy()
          loss_m=mse_loss.detach().cpu().numpy()

          loss_list.append(loss_np)
          if i%10==0:
               print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}".format(i,loss_np,loss_m,loss_p))
               save_image(synth_img.clamp(0,1),"save_image/encode1/{}.png".format(i))
               #np.save("loss_list.npy",loss_list)
               np.save("latent_W/{}.npy".format(name),dlatent.detach().cpu().numpy())

          






          



def caluclate_loss(synth_img,img,perceptual_net,img_p,MSE_Loss,upsample2d):
     #calculate MSE Loss
     mse_loss=MSE_Loss(synth_img,img) # (lamda_mse/N)*||G(w)-I||^2

     #calculate Perceptual Loss
     real_0,real_1,real_2,real_3=perceptual_net(img_p)
     synth_p=upsample2d(synth_img) #(1,3,256,256)
     synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)

     perceptual_loss=0
     perceptual_loss+=MSE_Loss(synth_0,real_0)
     perceptual_loss+=MSE_Loss(synth_1,real_1)
     perceptual_loss+=MSE_Loss(synth_2,real_2)
     perceptual_loss+=MSE_Loss(synth_3,real_3)

     return mse_loss,perceptual_loss




     







     

    

if __name__ == "__main__":
    main()



