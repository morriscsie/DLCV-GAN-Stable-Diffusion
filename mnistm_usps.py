import os
import sys
import torch.backends.cudnn as cudnn
import torch
from model import MNISTmodel
from train import train_dann
from utils import init_model, init_random_seed
import random
from dataset import get_mnistm,get_usps

class Config(object):
   
    finetune_flag = False
    lr_adjust_flag = 'simple'
    src_only_flag = False

    # params for datasets and data loader
    batch_size = 128

    # params for source dataset
    src_dataset = "mnistm"
    src_image_root = os.path.join('./hw2_data/digits/',src_dataset)
    print(f"src_path{src_image_root}")
   
   
    # params for target dataset
    tgt_dataset = "usps"
    tgt_image_root = os.path.join('./hw2_data/digits/',tgt_dataset)
    print(f"tgt_path{tgt_image_root}")
   
  

    # params for training dann
    gpu_id = '0'

    ## for digit
    num_epochs = 150
    manual_seed = 999
    alpha = 0

    # params for optimizing models
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-6

params = Config()
device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")

# init random seed
init_random_seed(params.manual_seed)

# load dataset
src_data_loader = get_mnistm(params.src_image_root, params.batch_size, mode="train")
src_data_loader_eval = get_mnistm(params.src_image_root, params.batch_size, mode="val")
tgt_data_loader = get_usps(params.tgt_image_root, params.batch_size, mode="train")
tgt_data_loader_eval = get_usps(params.tgt_image_root, params.batch_size, mode="val")
PATH = "./model/best_model_P3_ministm_usps.pt"
# load dann model
dann = init_model(net=MNISTmodel(), restore=None)

# train dann model
print("Training dann model")
if not (dann.restored and params.dann_restore):
    print("Restart!!!")
    dann = train_dann(dann, params, src_data_loader, tgt_data_loader, src_data_loader_eval, tgt_data_loader_eval, device, PATH)
