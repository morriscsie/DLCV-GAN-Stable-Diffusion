#ref:https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#fid=29.74 face_reg=89.7% epoch=100 nz=100
import argparse
import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from model import Generator,Discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#def main(args):
if __name__ == "__main__":
    # Set random seed for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    PATH = "./model/best_model_P1A.pt"
    save_path = "./test/"
    # Root directory for dataset
    dataroot = "hw2_data/face"
    # Number of workers for dataloader
    workers = 2
    # Batch size during training
    batch_size = 128
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64
    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    num_epochs = 100
    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    # Create the dataset
  
    dataset = dset.ImageFolder(root=dataroot,transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
   
    # # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.savefig("./test.png")
   
    # Create the generator
    netG = Generator(ngf, nz, nc).to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    print(netG)
    
    # Create the Discriminator
    netD = Discriminator(ndf, nc).to(device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    # Print the model
    print(netD)


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(1000, nz, 1, 1, device=device)
    
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    transform = transforms.Compose([
            transforms.ToPILImage(),
        ])
    # Training Loop
    iters = 0
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        # For each batch in the dataloader
        train_loop = tqdm(dataloader,position=0,leave=False,ncols=60,colour='green',desc='Epoch '+str(epoch+1))
        for i, data in enumerate(train_loop, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        
            iters += 1
        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'% (epoch, num_epochs,errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    torch.save({
        'epoch':(epoch+1),
        'netG_state_dict':netG.state_dict(),
        'netD_state_dict':netD.state_dict(),
        }, PATH)
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        #print(fake[0]) (1000,3,64,64)
        for j in range(1000):
            path = os.path.join(save_path,f"{j}.png")
            img = fake[j] / 2 + 0.5
            im = transform(img)
            im.save(path)
              

