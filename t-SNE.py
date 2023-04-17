import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


import random
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import manifold
  
if __name__ == '__main__':
  
    valid_dir = "./hw1_data/p1_data/val_50/"
    #PATH = "./model/best_model_P1B.pt"
    PATH = "./15epoch_model.pt"
    input_size = 224
    valid_transform =  transforms.Compose(
        [transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
   
    valid_dataset = P1_Train_Dataset(valid_dir,transform=valid_transform)
  

    batch_size = 128
    val_dataloader = DataLoader(valid_dataset, batch_size, shuffle = False, num_workers = 4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Resnet152()
    ckpt = torch.load(PATH)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.model_ft
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(model)
    model = model.to(device)
    validate_loop = tqdm(val_dataloader,position=0,leave=False,ncols=60,colour='green')
    outputs = []
    labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            imgs,label = batch
            imgs = imgs.to(device)
            label = label.to(device)
            output = model(imgs)
            output = output.view(-1,2048)
            output = output.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            outputs.append(output)
            labels.append(label)
    #t-SNE
    x = np.concatenate(outputs,axis=0)
    y = np.concatenate(labels,axis=0)
    print(x.shape)
    print(y.shape)
    x_scaled = x
    X_tsne = manifold.TSNE(n_components=2,init='random',verbose=1).fit_transform(x_scaled)
    
   
    df = pd.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=y))
    df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')
    print('Shape after t-SNE: ', X_tsne.shape)
    plt.title('t-SNE Graph')
    plt.show()
    plt.savefig("./t-SNE.jpg")