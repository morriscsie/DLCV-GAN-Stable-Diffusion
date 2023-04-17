import os
import shutil
import pandas as pd
from PIL import Image
def organize_data(path,type):
    os.makedirs(os.path.join(path,type),exist_ok=True)
    df = pd.read_csv(os.path.join(path,type+".csv"))
    for i in range(len(df)):
        img_name = df.iloc[i,0]
        img_path = os.path.join(path,"data",img_name)
        img = Image.open(img_path).convert("RGB")
        img.save( os.path.join(path,type,img_name))
if __name__=='__main__':
    #organize training data
    test_path = "./hw2_data/digits/svhn/"
    type = "val"
    #val_path = "./hw2_data/digits/usps/"
    organize_data(test_path,type)
    #organize_data(val_path)