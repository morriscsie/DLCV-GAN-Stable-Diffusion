from argparse import ArgumentParser
from pathlib import Path
import os
import random
import torch
import torchvision.transforms as transforms
from model import Generator,Discriminator

def main(args):   
    # Set random seed for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 300
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
 
    num_generate_ = 5000
    num_result = 20000
    ckpt = torch.load(args.ckpt_path)
    netG = Generator(ngf, nz, nc)
    netD = Discriminator(ndf, nc)
    netG.load_state_dict(ckpt["netG_state_dict"])
    netD.load_state_dict(ckpt["netD_state_dict"])
    netG.to(device)
    netD.to(device)
    with torch.no_grad():
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        img_prob = []
        for _ in range(num_result//num_generate_):
            fixed_noise = torch.randn(num_generate_, nz, 1, 1, device=device)
            fake = netG(fixed_noise)
            prob = netD(fake)
            fake = fake.detach().cpu()
            prob = prob.detach().cpu()
            for j in range(num_generate_):
                img = fake[j] / 2 + 0.5
                im = transform(img)
                img_prob.append((im,prob[j].item()))
               
        img_prob.sort(key=lambda tup: tup[1],reverse=True)  # sorts in place
       
        for j in range(1000):
            path = os.path.join(args.generated_dir,f"{j}.png")
            im =  img_prob[j][0]
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
        default="./model/best_model_P2.pt"
    )
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)

