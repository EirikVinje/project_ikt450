 import os
import tarfile
import csv

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib as plt
from PIL import Image
import cv2
import pickle


#  ['id', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax', 'Pneumoperitoneum', 'Pneumomediastinum', 'Subcutaneous Emphysema', 'Tortuous Aorta', 'Calcification of the Aorta', 'No Finding', 'subj_id']


def load_labels(batch_x, dataset):

    path_to_labels = f"/home/eirikmv/data/chestxray8/labels/{dataset}_labels.csv" 
    labels = pd.read_csv(path_to_labels)

    all_x = []
    all_y = []
    
    print("reading labels...")
    for i in tqdm(range(len(batch_x)):

        x, filename = batch_x[i]

        label = labels[labels["id"] == filename]

        if not label.empty:

            label = np.array(label.iloc[0].values[1:-1])
            label = np.where(label == 1)[0][0]
    
            if label != 19:
    
                all_x.append(x)
                all_y.append(label)
            
    return all_x, all_y


def load_images(batch, x_shape:tuple = (512, 512)):

    path_to_batch = f"/home/eirikmv/data/chestxray8/batch_{batch}/images"
    
    batch_files_orig = os.listdir(path_to_batch)

    batch_x = []
    
    nonetype = 0
    dim3 = 0

    print(f"reading images from batch {batch}...")
    for file in tqdm(batch_files_orig):

        path_to_image = os.path.join(path_to_batch, file)
        img = cv2.imread(path_to_image)
        
        if type(img) == type(None):
            nonetype += 1
            continue    

        img = cv2.resize(img, x_shape, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        img = cv2.resize(img, x_shape)
        
        batch_x.append([img, file])

    
    print("****************")
    print("nonetype in x:", nonetype)
    print("size batch:", len(batch_x))
    print("****************")
    
    return batch_x


def load_data():

    datasets = ["train", "test", "validation"]

    data = {"train" : [[], []], "test" : [[], []], "validation" : [[], []]}

    for i in range(0, 1):

        batch_x = load_images(i)
        for settype in datasets:
        
            x, y = load_labels(batch_x, settype)
        
            print(f"batch {i} {settype} size : {len(x)}")
            
            data[settype][0].extend(x)
            data[settype][1].extend(y)

    print("train size", len(data["train"][0]))
    print("test size", len(data["test"][0]))
    print("validation size", len(data["validation"][0]))
    
    for settype in datasets:
        pickle.dump(data[settype], open(f"/home/eirikmv/data/chestxray8/dataset/{settype}.pkl", "wb"))


if __name__ == "__main__":

    load_data()        

    



















