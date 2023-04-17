#ref:https://github.com/wogong/pytorch-dann?fbclid=IwAR3jypcJ5JxQbkkvxfnEyUkOawl1rxJG_c8o-TRxfOGwNbaxB9SKFLpGjE0
import os
import sys
import torch
from argparse import ArgumentParser
from pathlib import Path
from model import MNISTmodel
from utils import init_model, init_random_seed
import random
from dataset import get_usps_test,get_svhn_test
from test import generate
def main(args):
    batch_size = 1
    if("svhn" in args.test_dir.split("/")):
        model_name = "best_model_P3_ministm_svhn.pt"
        # load dataset
        tgt_data_loader_test = get_svhn_test(args.test_dir, batch_size)
    else:
        model_name = "best_model_P3_ministm_usps.pt"   
        # load dataset
        tgt_data_loader_test = get_usps_test(args.test_dir, batch_size)
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   
    # train dann model
    print("Testing dann model")
    # load dann model
    dann = init_model(net=MNISTmodel(), restore=None)
    ckpt = torch.load(os.path.join(args.ckpt_path,model_name))
    dann.load_state_dict(ckpt["model_state_dict"])
    dann.to(device)

    path_label  = generate(dann, tgt_data_loader_test, device, flag='target')
  
    with open(args.pred_file, "w") as f:
        f.write("image_name,label\n")
        l = len(path_label)
        for i in range(l):
            f.write(f"{path_label[i][0]},{path_label[i][1]}\n")
    f.close()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--test_dir",
        type=str,
        help="Path to the test images dir.",
        required=True
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Path to the pred file.",
        required=True
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the ckpt.pt.",
        default="./model/best_model_P3_ministm_usps.pt"
    )
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)


   


