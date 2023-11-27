import os
import gc

import pickle
import torch
import optuna
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from torchmetrics.classification import MultilabelAccuracy

from mobilenetv1 import MobileNetV1

class Xray_dataset(Dataset):
    
    def __init__(self, images, labels, test : bool):

        self.images = images
        self.labels = labels
        self.test = test

        self.labels = torch.from_numpy(np.array(self.labels).astype(np.int32))
        
        self.transform_train = transforms.Compose([transforms.ToTensor(),
                                                   transforms.RandomRotation(20),
                                                   transforms.RandomHorizontalFlip()])

        self.transform_test = transforms.Compose([transforms.ToTensor()])

        self.transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
            return len(self.images)
    
    def __getitem__(self, idx):
        
        img = self.images[idx]
        label = self.labels[idx]

        if not self.test:
            img = self.transform_train(img)

        elif self.test:
            img = self.transform_test(img)
        
        return img, label
    

def train(epochs, batch_size, lr):

    os.system("rm -f model/model.pt")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_channels = 3
    
    data_dir = "/home/eirikmv/data/chestxray8/dataset"
    
    train_x, train_y = pickle.load(open(os.path.join(data_dir, "train.pkl"), "rb"))
    val_x, val_y = pickle.load(open(os.path.join(data_dir, "validation.pkl"), "rb"))

    print(f"train size: {len(train_y)}") 
    print(f"val size: {len(val_y)}")
    
    train_dataloader = DataLoader(Xray_dataset(train_x, train_y, test = False), 
                                  batch_size=batch_size, 
                                  shuffle=True)
    
    val_dataloader = DataLoader(Xray_dataset(val_x, val_y, test = False), 
                                batch_size=batch_size, 
                                shuffle=True)

    model = MobileNetV1(len(train_y[0]), n_channels).to(device)
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    all_loss = []
    all_train_acc = []
    all_val_acc = []
    
    print("Training model....")
    for epoch in range(epochs):

        for (x, y) in tqdm(train_dataloader):
            
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            
            yhat = model(x)
            
            loss = criterion(yhat, y.float())

            loss.backward()    
            optimizer.step()
        
        lr_scheduler.step()
        
        print(f"Epoch:{epoch} Loss:{loss.item()}")
        

    torch.save(model, "model/model.pt")
    
if __name__ == "__main__":

    epochs = 5
    batch_size = 64
    lr = 0.001
    
    train(epochs, batch_size, lr)





