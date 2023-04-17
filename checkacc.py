import os
import shutil
import pandas as pd
from PIL import Image
  
if __name__=='__main__':
   
  
    path = "./hw2_data/digits/svhn/"
    type = "val"
    df1 = pd.read_csv(os.path.join(path,type+".csv"))
    df2 = pd.read_csv(os.path.join("./test.csv"))
    acc = 0
    if (len(df1)==len(df2)):
        print("alright!")
    for i in range(len(df1)):
        if df2.iloc[i,1] == df1.iloc[i,1]:
            acc += 1
    print("acc={:.4f}".format(acc/len(df1)))
    #organize_data(val_path)