import os
import sys
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from argparse import ArgumentParser
from pathlib import Path
sys.path.append('../')
from model import MNISTmodel,MNISTmodelt
from train import train_dann
from utils import init_model, init_random_seed
import random
from dataset import get_usps,get_svhn,get_mnistm
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
import pandas as pd 
def tsne(model, src_data_loader_eval, tgt_data_loader_eval, device):
    """Evaluate model for dataset."""
    # set eval state for Dropout and BN layers
    model.eval()

    # init loss and accuracy
    loss_ = 0.0
    acc_ = 0.0
    acc_domain_ = 0.0
    n_total = 0
 
    # set loss function
    criterion = nn.CrossEntropyLoss()
    outputs = []
    Labels = []
    domain_label_outputs = []
    # tgt
    for (images, labels) in tgt_data_loader_eval:
        images = images.to(device)
        labels = labels.to(device)  #labels = labels.squeeze(1)
        size = len(labels)
        labels_domain = torch.ones(size).long().to(device)

        preds, domain, feature = model(images, alpha=0)
       
        loss_ += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        pred_domain = domain.data.max(1)[1]
        acc_ += pred_cls.eq(labels.data).sum().item()
        acc_domain_ += pred_domain.eq(labels_domain.data).sum().item()
        n_total += size
        feature = feature.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        labels_domain = labels_domain.detach().cpu().numpy()
        outputs.append(feature)
        Labels.append(labels)
        domain_label_outputs.append(labels_domain)
    loss = loss_ / n_total
    acc = acc_ / n_total
    acc_domain = acc_domain_ / n_total
    
    # src
    for (images, labels) in src_data_loader_eval:
        images = images.to(device)
        labels = labels.to(device)  #labels = labels.squeeze(1)
        size = len(labels)
        labels_domain = torch.zeros(size).long().to(device)

        preds, domain, feature = model(images, alpha=0)
       
    

        pred_cls = preds.data.max(1)[1]
        pred_domain = domain.data.max(1)[1]
        acc_ += pred_cls.eq(labels.data).sum().item()
        acc_domain_ += pred_domain.eq(labels_domain.data).sum().item()
        n_total += size
        feature = feature.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        labels_domain = labels_domain.detach().cpu().numpy()
        outputs.append(feature)
        Labels.append(labels)
        domain_label_outputs.append(labels_domain)
    

    #t-SNE
    x = np.concatenate(outputs,axis=0)
    y = np.concatenate(Labels,axis=0)
    z = np.concatenate(domain_label_outputs,axis=0)
    print(x.shape)
    print(y.shape)
    print(z.shape)
    x_scaled = x
    X_tsne = manifold.TSNE(n_components=2,init='random',verbose=1).fit_transform(x_scaled)
    
   
    df = pd.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=y))
    df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')
    print('Shape after t-SNE: ', X_tsne.shape)
    plt.title('t-SNE Graph')
    plt.show()
    plt.savefig("./t-SNE.jpg")

    df = pd.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=z))
    df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')
    print('Shape after t-SNE: ', X_tsne.shape)
    plt.title('t-SNE Graph')
    plt.show()
    plt.savefig("./t-SNEdomain.jpg")

    return loss, acc, acc_domain
def main():

        #  for datasets and data loader
        batch_size = 128
        # params for source dataset
        src_dataset = "mnistm"
        src_image_root = os.path.join('./hw2_data/digits/',src_dataset)
        print(f"src_path{src_image_root}")

        # params for target dataset
        tgt_dataset = "usps" #"svhn"
        tgt_image_root = os.path.join('./hw2_data/digits/',tgt_dataset)
        print(f"tgt_path{tgt_image_root}")
    
    
        #svhn
        #model_name = "./model/best_model_P3_ministm_svhn.pt"
        #usps
        model_name = "./model/best_model_P3_ministm_usps.pt"   
            
       
  
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   
        # train dann model
        print("Testing dann model")
        # load dann model
        dann = init_model(net=MNISTmodelt(), restore=None)
        ckpt = torch.load(os.path.join(model_name))
        dann.load_state_dict(ckpt["model_state_dict"])
        dann.to(device)
    
        src_data_loader_eval = get_mnistm(src_image_root, batch_size, mode="val")
        tgt_data_loader_eval = get_usps(tgt_image_root, batch_size, mode="val")
        #tgt_data_loader_eval = get_svhn(tgt_image_root, batch_size, mode="val")
        tgt_test_loss, tgt_acc, tgt_acc_domain = tsne(dann, src_data_loader_eval, tgt_data_loader_eval, device)
        
        print("tgt_acc={:.2f}%".format(tgt_acc*100))
   

# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument(
#         "--test_dir",
#         type=str,
#         help="Path to the test images dir.",
#         required=True
#     )
#     parser.add_argument(
#         "--pred_file",
#         type=Path,
#         help="Path to the pred file.",
#         required=True
#     )
#     parser.add_argument(
#         "--ckpt_path",
#         type=Path,
#         help="Path to the ckpt.pt.",
#         default="./model/best_model_P3_ministm_usps.pt"
#     )
#     args = parser.parse_args()
#     return args
if __name__ == '__main__':
    #args = parse_args()
    main()


   


