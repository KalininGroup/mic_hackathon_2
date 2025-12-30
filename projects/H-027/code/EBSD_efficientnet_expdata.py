# Code modified from https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn import metrics
import shutil
import argparse
import pickle
from torch.utils.data import Dataset
from efficientnet_pytorch import EfficientNet
from collections import defaultdict
import cv2

cudnn.benchmark = True
plt.ion()   # interactive mode

projectdir='/projects/b1190/ayz0064'

print('is gpu available: ',torch.cuda.is_available())
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=4)
parser.add_argument('-v', '--validation',action="store_true")
parser.add_argument('--clahe',action="store_true")
parser.add_argument('--modelpath',type=str)
args = parser.parse_args()

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
test_transform= transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()
    img = Image.open(img_path).convert('RGB')
    img=test_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        model.train(mode=was_training)
    return phaselist[preds[0]]

phaselist=['Ni','FeNi3','Ni3Al']
if args.validation:
    with open('/projects/b1190/ayz0064/EBSD_ad/datadict_val.pkl','rb') as file:
        datadict = pickle.load(file)
else:
     with open('/projects/b1190/ayz0064/EBSD_ad/datadict.pkl','rb') as file:
        datadict = pickle.load(file)

testimgs=[]
testlabels=[]
for img in datadict['test']:
    imgname=img.split('/')[-1]
    if 'FeAl' in imgname:
        comp='FeAl'
    else:
        comp=imgname.replace('originalsize_','').split('_')[0].split(' ')[0]
    if comp not in ['FeNi3','Ni','Ni3Al']:
        continue
    label=phaselist.index(comp)
    testimgs.append(img)
    testlabels.append(label)
if args.validation:
    valimgs=[]
    vallabels=[]
    for img in datadict['val']:
        imgname=img.split('/')[-1]
        if 'FeAl' in imgname:
            comp='FeAl'
        else:
            comp=imgname.replace('originalsize_','').split('_')[0].split(' ')[0]
        if comp not in ['Ni','FeNi3','Ni3Al']:
            continue
        label=phaselist.index(comp)
        valimgs.append(img)
        vallabels.append(label)
valresults={epoch:{} for epoch in range(args.epochs)}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train']:
                if phase=='train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase=='train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase=='train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if True: 
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
            if args.validation:
                for valimg in valimgs:
                    pred=visualize_model_predictions(model,valimg)
                    valresults[epoch][valimg]=pred
            else:    
                for testimg in testimgs:
                    pred=visualize_model_predictions(model,testimg)
                    valresults[epoch][testimg]=pred

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

import random
from torchvision.io import read_image
from PIL import Image
from torchvision.transforms import ToTensor

for i in [0]:
    imgs=[]
    labels=[]
    phasecount=defaultdict(int)
    import json
    clahe = cv2.createCLAHE(clipLimit=8,tileGridSize=(4,4))
    with open('/projects/b1190/ayz0064/EBSD_ad/datadict_val.pkl','rb') as file:
        datadict = pickle.load(file)
    if not args.validation:
        trainsplits=[datadict['train'],datadict['val']]
    else:
        trainsplits=[datadict['train']]
    for split in trainsplits:
        for img in split:
            imgname=img.split('/')[-1]
            if 'FeAl' in imgname:
                comp='FeAl'
            else:
                comp=imgname.replace('originalsize_','').split('_')[0].split(' ')[0]
            if comp not in ['FeNi3','Ni','Ni3Al']:
                continue
            label=phaselist.index(comp)
            imgs.append(img)
            labels.append(label)

    n_classes=len(list(set(labels)))
    class CustomImageDataset(Dataset):
        def __init__(self, img_list, label_list, transform=None):
            self.transform = transform
            self.img_list=img_list
            self.label_list=label_list
        def __len__(self):
            return len(self.img_list)
        def __getitem__(self, idx):
            if not args.clahe:
                image = Image.open(self.img_list[idx]).convert('RGB')
            #img=read_image(self.img_list[idx])
            if args.clahe:
                image = np.array(Image.open(self.img_list[idx]))
                #image=clahe.apply(np.array(image))
                image=Image.fromarray(image).convert('RGB')
            label = self.label_list[idx]
            image = self.transform(image)
            return image, label
    image_datasets = {'train': CustomImageDataset(imgs,labels,transform=data_transforms['train'])}
    dataset_sizes = {'train': len(image_datasets['train'])}
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=4)}
    del(imgs)
    del(labels)
    model_ft=EfficientNet.from_pretrained('efficientnet-b4')
    model_ft._fc = torch.nn.Linear(model_ft._fc.in_features, n_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=args.epochs)

    import pickle
    if args.validation:
        with open('/home/ayz0064/crossentropy_preds_val.pkl', 'wb') as f:
            pickle.dump(valresults, f)
    else:
        with open('/home/ayz0064/crossentropy_preds_test.pkl', 'wb') as f:
            pickle.dump(valresults, f)
