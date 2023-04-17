#ref:https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#nz=100 epoch=100 fid=34 face_reg=80.5%
#nz=100 epoch=100 fid=29 face_reg=86.9% #save Dx higher
#test fid=52.34 face_reg=84.7% original unnormalize
#test fid=51.57 face_reg=88.7% easy unnormalize
#test fid=61.54 face_reg=82.9% transform unnormalize
#test fid=34 face_reg=77.7% easy unnormalize
#test fid=26.87 face_reg=89.8% nz=300 epoch=300 1000/1000
from argparse import ArgumentParser
from pathlib import Path
import os
import random
import torch
import torchvision.transforms as transforms
from model import Generator,Discriminator
import numpy as np

def main(args):   
    # Set random seed for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create the generator
    netG = Generator(ngf, nz, nc).to(device)
    # Print the model
    print(netG)
    # Create the Discriminator
    netD = Discriminator(ndf, nc).to(device)
    # Print the model
    print(netD)
    transform = transforms.Compose([
            transforms.ToPILImage(),
        ])
 
    
    ckpt = torch.load(args.ckpt_path)
    netG = Generator(ngf, nz, nc)
    netD = Discriminator(ndf, nc)
    netG.load_state_dict(ckpt["netG_state_dict"])
    netD.load_state_dict(ckpt["netD_state_dict"])
    netG.to(device)
    netD.to(device)
    fixed_noise = torch.randn(1000, nz, 1, 1, device=device)

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        #print(fake[0]) (1000,3,64,64)
        for j in range(1000):
            path = os.path.join(args.generated_dir,f"{j}.png")
            img = fake[j] / 2 + 0.5
            im = transform(img)
            im.save(path)
      
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--generated_dir",
        type=Path,
        help="Path to the generated dir.",
        required=True
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the ckpt.pt.",
        default="./model/best_model_P1A.pt"
    )
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
