import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pyplot as plt 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from stylegan_layers import  G_mapping,G_synthesis,D_basic


resolution=1024

g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(resolution=resolution))    
]))

d_basic = D_basic(resolution=resolution)
a=True




tensorflow_dir="weight_files/tensorflow/"
pytorch_dir="weight_files/pytorch/"
weight_name="karras2019stylegan-ffhq-1024x1024"

if a:
    # this can be run to get the weights, but you need the reference implementation and weights
    import dnnlib, dnnlib.tflib, pickle, torch, collections
    dnnlib.tflib.init_tf()
    weights = pickle.load(open(tensorflow_dir+weight_name+".pkl",'rb'))
    weights_pt = [collections.OrderedDict([(k, torch.from_numpy(v.value().eval())) for k,v in w.trainables.items()]) for w in weights]
    torch.save(weights_pt, pytorch_dir+weight_name+".pt")
if a:
    # then on the PyTorch side run
    state_G, state_D, state_Gs = torch.load(pytorch_dir+weight_name+".pt")
    def key_translate(k):
        k = k.lower().split('/')
        if k[0] == 'g_synthesis':
            if not k[1].startswith('torgb'):
                k.insert(1, 'blocks')
        k = '.'.join(k)
        k = (k.replace('const.const','const').replace('const.bias','bias').replace('const.stylemod','epi1.style_mod.lin')
              .replace('const.noise.weight','epi1.top_epi.noise.weight')
              .replace('conv.noise.weight','epi2.top_epi.noise.weight')
              .replace('conv.stylemod','epi2.style_mod.lin')
              .replace('conv0_up.noise.weight', 'epi1.top_epi.noise.weight')
              .replace('conv0_up.stylemod','epi1.style_mod.lin')
              .replace('conv1.noise.weight', 'epi2.top_epi.noise.weight')
              .replace('conv1.stylemod','epi2.style_mod.lin')
              .replace('torgb_lod0','torgb')
              .replace('fromrgb_lod0','fromrgb'))
        if 'torgb_lod' in k or 'fromrgb_lod' in k: # we don't want the lower layers to/from RGB
            k = None
        return k

    def weight_translate(k, w):
        k = key_translate(k)
        if k.endswith('.weight'):
            if w.dim() == 2:
                w = w.t()
            elif w.dim() == 1:
                pass
            else:
                assert w.dim() == 4
                w = w.permute(3, 2, 0, 1)
        return w

if a:
    param_dict = {key_translate(k) : weight_translate(k, v) for k,v in state_Gs.items() if key_translate(k) is not None}
    if a:
        sd_shapes = {k : v.shape for k,v in g_all.state_dict().items()}
        param_shapes = {k : v.shape for k,v in param_dict.items() }

        for k in list(sd_shapes)+list(param_shapes):
            pds = param_shapes.get(k)
            sds = sd_shapes.get(k)
            if pds is None:
                print ("sd only", k, sds)
            elif sds is None:
                print ("pd only", k, pds)
            elif sds != pds:
                print ("mismatch!", k, pds, sds)

    g_all.load_state_dict(param_dict, strict=False) # needed for the blur kernels
    torch.save(g_all.state_dict(), pytorch_dir+weight_name+".pt")
if a:
    param_dict = {key_translate(k) : weight_translate(k, v) for k,v in state_D.items() if key_translate(k) is not None}
    if a:
        sd_shapes = {k : v.shape for k,v in d_basic.state_dict().items()}
        param_shapes = {k : v.shape for k,v in param_dict.items() }

        for k in list(sd_shapes)+list(param_shapes):
            pds = param_shapes.get(k)
            sds = sd_shapes.get(k)
            if pds is None:
                print ("sd only", k, sds)
            elif sds is None:
                print ("pd only", k, pds)
            elif sds != pds:
                print ("mismatch!", k, pds, sds)

    d_basic.load_state_dict(param_dict, strict=False) # needed for the blur kernels
    torch.save(d_basic.state_dict(), pytorch_dir+weight_name+"_d.pt")