import numpy as np 
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
    parser.add_argument('--weight_file',default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt",type=str)
    parser.add_argument('--latent_file1',default="latent_W/0.npy")
    parser.add_argument('--latent_file2',default="latent_W/sample.npy")




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


    latents_0=np.load(args.latent_file1)
    latents_1=np.load(args.latent_file2)



    latents_0=torch.tensor(latents_0).to(device)
    latents_1=torch.tensor(latents_1).to(device)



    for i in range(100):
        alpha=(1/100)*i
        latents=alpha*latents_0+(1-alpha)*latents_1

        synth_img=g_synthesis(latents)
        synth_img = (synth_img + 1.0) / 2.0
        save_image(synth_img.clamp(0,1),"morph_result/encode1/{}.png".format(i))

     

    

if __name__ == "__main__":
    main()




