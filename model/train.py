import os
import gc

import pickle
import torch
import optuna
from torchvision import transforms
from torch.utils.data import Dataset

from mobilenetv1 import MobileNetV1


class Xray_dataset(Dataset):
    
    def __init__(self, img, label, test : bool):

        self.img = img
        self.label = label
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test = test
        
        self.transform_train = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(20),
                                    transforms.RandomHorizontalFlip()])

        self.transform_test = transforms.Compose([transforms.ToTensor()])

        self.transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
            return len(self.img)
    
    def __getitem__(self, idx):
        
        img = self.img[idx]
        label = self.label[idx]

        if not self.test:
            img = self.transform_train(img)

        elif self.test:
            img = self.transform_test(img)
        
        return img, label


def accuracy(dataloader, model):

    with torch.no_grad():

        total = 0
        correct = 0
        
        for x, y in enumerate(dataloader):
            
            predicted = model(x)
            predicted = torch.argmax(predicted, dim=1)

            correct += (predicted==y).sum().item()
            total += predicted.shape[0]
        
        return round(correct/total, 2)


def train(epochs, batch_size, lr):

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.empty_cache()
    gc.collect()

    train_x, train_y = pickle.load(open("smalldata/train.pkl", "rb"))
    val_x, val_y = pickle.load(open("smalldata/validation.pkl", "rb"))

    train_dataloader = torch.utils.data.DataLoader(Xray_dataset(train_x, train_y, test = False), batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(Xray_dataset(val_x, val_y, test = False), batch_size=batch_size, shuffle=True)

    model = MobileNetV1().to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    all_loss = []
    all_train_acc = []
    all_val_acc = []
    
    print("Training model....")
    for epoch in range(epochs):

        for i, (x, y) in enumerate(train_dataloader):
            
            optimizer.zero_grad()
            
            yhat = model(x)
            loss = criterion(yhat, y)

            loss.backward()    
            optimizer.step()

        lr_scheduler.step()

        new_loss = loss.item()
        train_acc = accuracy(train_dataloader, model)
        val_acc = accuracy(val_dataloader, model)
        
        all_loss.append(new_loss)
        all_train_acc.append()

        print(f"Epoch:{epoch} Loss:{new_loss} train accuracy:{train_acc} val accuracy {val_acc}")


    test_x, test_y = pickle.load(open("smalldata/test.pkl", "rb"))
    test_dataloader = torch.utils.data.DataLoader(Xray_dataset(test_x, test_y, test = True), batch_size=batch_size, shuffle=True)
    
    acc = accuracy(test_dataloader, model)

    print(f"Test accuracy:{acc}")


if __name__ == "__main__":

    epochs = 10
    batch_size = 16
    lr = 0.001
    
    train(epochs, batch_size, lr)





