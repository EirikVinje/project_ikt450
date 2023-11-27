import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelPrecision
import pickle
import os
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

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


def accuracy_set(model, dataloader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():

        all_acc = 0
        all_pre = 0
        count = 0
        
        for (x, y) in tqdm(dataloader):

            x = x.to(device)
            y = y.to(device)
            
            predicted = model(x)

            # print(predicted)
            # print(predicted.shape)

            # assert False
            
            acc = MultilabelAccuracy(num_labels=19).to(device)(predicted.to(device), y)
            pre = MultilabelPrecision(num_labels=19).to(device)(predicted.to(device), y)

            count += 1
            all_acc += acc
            all_pre += pre
            
        total_acc = all_acc/count
        total_pre = all_pre/count

        return total_acc, total_pre



def evaluate_model(batch_size):
    
    model = torch.load("model/model.pt")

    data_dir = "/home/eirikmv/data/chestxray8/dataset"
    
    test_x, test_y = pickle.load(open(os.path.join(data_dir, "test.pkl"), "rb"))
    train_x, train_y = pickle.load(open(os.path.join(data_dir, "train.pkl"), "rb"))
    val_x, val_y = pickle.load(open(os.path.join(data_dir, "validation.pkl"), "rb"))

    train_dataloader = DataLoader(Xray_dataset(train_x, train_y, test = False), 
                                  batch_size=batch_size, 
                                  shuffle=True)
    
    val_dataloader = DataLoader(Xray_dataset(val_x, val_y, test = False), 
                                batch_size=batch_size, 
                                shuffle=True)
    
    test_dataloader = DataLoader(Xray_dataset(test_x, test_y, test = True), 
                                                  batch_size=batch_size, 
                                                  shuffle=True)


    train_acc, train_pre = accuracy_set(model, train_dataloader)
    val_acc, val_pre = accuracy_set(model, val_dataloader)
    test_acc, test_pre = accuracy_set(model, test_dataloader)

    print("train:", train_acc.item(), "|", train_pre)
    print("val", val_acc.item(), "|", val_pre)
    print("test", test_acc.item(), "|", test_pre)


if __name__ == "__main__":

    batch_size = 64
    
    evaluate_model(batch_size)










