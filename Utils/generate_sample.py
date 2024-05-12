import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import torch
from torchvision import utils
from Models.StyleGan2.model import Generator
from tqdm import tqdm
import einops



def generate(args, g_ema,  device, mean_latent):
	with torch.no_grad():
		g_ema.eval()
		for i in tqdm(range(args.pics)):
			latent1=torch.randn(18,512)
			latent1=latent1.unsqueeze(0)
			latent1=einops.repeat(latent1, 'b h w ->(repeat b) h w',repeat=8)
			#latent2=torch.randn(18,812)
			#latent=torch.cat((latent_1[:index], latent_2[index:]), 0).unsqueeze(0)
			sample_z = latent1.to(device)
			sample, latents = g_ema([sample_z], input_is_latent=False, return_latents=True)
        
			utils.save_image(sample,f"sample/{str(i).zfill(6)}.png",nrow=1,normalize=True,range=(-1, 1))
                	

if __name__ == "__main__":
    device = "cuda"
    
    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    
    parser.add_argument("--size", type=int, default=1024, help="output image size of the generator" )
    
    parser.add_argument("--sample",type=int,default=1,help="number of samples to be generated for each image")
        
    
    parser.add_argument("--pics", type=int, default=20, help="number of images to be generated")

    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
   
    parser.add_argument("--truncation_mean",type=int,default=4096, help="number of vectors to calculate mean for the truncation")
    
    parser.add_argument("--ckpt",type=str,default="/home/user1/aravind/New_ID/stylegan2-ffhq-config-f.pt",help="path to the model 		checkpoint")
   
    parser.add_argument("--channel_multiplier",type=int,default=2,help="channel multiplier of the generator. config-f = 2, else = 1")
    parser.add_argument("--batch_size",type=int,default=4,help='batch_size')
   
    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8
    
    
    g_ema = Generator(args.size, args.latent, args.n_mlp,channel_multiplier=args.channel_multiplier).to(device)
    
        
    
    checkpoint = torch.load(args.ckpt)
    
    g_ema.load_state_dict(checkpoint["g_ema"])
    
    
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
    	mean_latent = None
    	
    generate(args, g_ema, device, mean_latent)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                	
