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
     parser.add_argument('--src_im1',default="source_image/sample.png")
     parser.add_argument('--src_im2',default="source_image/0.png")
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


     img_0=image_reader(args.src_im1) #(1,3,1024,1024) -1~1
     img_0=img_0.to(device)

     img_1=image_reader(args.src_im2)
     img_1=img_1.to(device) #(1,3,1024,1024)


     MSE_Loss=nn.MSELoss(reduction="mean")
     upsample2d=torch.nn.Upsample(scale_factor=0.5, mode='bilinear')

     img_p0=img_0.clone() #resize for perceptual net
     img_p0=upsample2d(img_p0)
     img_p0=upsample2d(img_p0) #(1,3,256,256)

     img_p1=img_1.clone()
     img_p1=upsample2d(img_p1)
     img_p1=upsample2d(img_p1) #(1,3,256,256)




     perceptual_net=VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device) #conv1_1,conv1_2,conv2_2,conv3_3
     dlatent_a=torch.zeros((1,18,512),requires_grad=True,device=device) #appearace latent s1
     dlatent_e=torch.zeros((1,18,512),requires_grad=True,device=device) # expression latent s2
     optimizer=optim.Adam({dlatent_a,dlatent_e},lr=0.01,betas=(0.9,0.999),eps=1e-8)

     alpha=torch.zeros((1,18,512)).to(device)
     alpha[:,3:5,:]=1

     print("Start")
     loss_list=[]
     for i in range(args.iteration):
          optimizer.zero_grad()
          synth_img_a=g_synthesis(dlatent_a)
          synth_img_a= (synth_img_a + 1.0) / 2.0

          synth_img_e=g_synthesis(dlatent_e)
          synth_img_e= (synth_img_e + 1.0) / 2.0

          loss_1=caluclate_contentloss(synth_img_a,perceptual_net,img_p1,MSE_Loss,upsample2d)
          loss_1.backward()

          optimizer.step()

          loss_2=caluclate_styleloss(synth_img_e,img_p0,perceptual_net,upsample2d)
          loss_2.backward()
          optimizer.step()

          loss_1=loss_1.detach().cpu().numpy()
          loss_2=loss_2.detach().cpu().numpy()



          dlatent1=dlatent_a*alpha+dlatent_e*(1-alpha)
          dlatent2=dlatent_a*(1-alpha)+dlatent_e*alpha

          synth_img1=g_synthesis(dlatent1)
          synth_img1= (synth_img1 + 1.0) / 2.0

          synth_img2=g_synthesis(dlatent2)
          synth_img2= (synth_img2 + 1.0) / 2.0

          if i%10==0:
            print("iter{}:   loss0 --{},  loss1 --{}".format(i,loss_1,loss_2))
            save_image(synth_img_a.clamp(0,1),"save_image/exchange/a/{}_a.png".format(i))
            save_image(synth_img_e.clamp(0,1),"save_image/exchange/e/{}_e.png".format(i))
            save_image(synth_img1.clamp(0,1),"save_image/exchange/result1/{}_exchange1.png".format(i))
            save_image(synth_img2.clamp(0,1),"save_image/exchange/result2/{}_exchange2.png".format(i))



            np.save("latent_W/exchange1.npy",dlatent1.detach().cpu().numpy())
            np.save("latent_W/exchange2.npy",dlatent2.detach().cpu().numpy())
          






          



def caluclate_contentloss(synth_img,perceptual_net,img_p,MSE_Loss,upsample2d): #W_l

     real_0,real_1,real_2,real_3=perceptual_net(img_p)
     synth_p=upsample2d(synth_img) #(1,3,256,256)
     synth_p=upsample2d(synth_p)
     synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)

     perceptual_loss=0


     perceptual_loss+=MSE_Loss(synth_0,real_0)
     perceptual_loss+=MSE_Loss(synth_1,real_1)

     perceptual_loss+=MSE_Loss(synth_2,real_2)
     perceptual_loss+=MSE_Loss(synth_3,real_3)



     return perceptual_loss



class StyleLoss(nn.Module):
     def __init__(self, target_feature):
          super(StyleLoss, self).__init__()
          self.target = self.gram_matrix(target_feature).detach()
     def forward(self, input):
          G = self.gram_matrix(input)
          self.loss = F.mse_loss(G, self.target)
          return self.loss
     def gram_matrix(self,input):
          a, b, c, d = input.size()  
          features = input.view(a * b, c * d)  

          G = torch.mm(features, features.t())  
          return G.div(a * b * c * d)




def caluclate_styleloss(synth_img,img_p,perceptual_net,upsample2d):

     synth_p=upsample2d(synth_img) #(1,3,256,256)
     synth_p=upsample2d(synth_p)

     _,_,_,style_real=perceptual_net(img_p) #conv3_3
     _,_,_,style_synth=perceptual_net(synth_p)

     style_loss=StyleLoss(style_real)

     loss=style_loss(style_synth)

     return loss

 



     



     

    

if __name__ == "__main__":
    main()



