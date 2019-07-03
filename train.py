import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, models, transforms
import numpy as np
import argparse
import copy
import sys
import os
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--arch', default="vgg19", type=str, choices=["vgg19", "alexnet"])
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--hidden_units', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu', default=True)
    return parser.parse_args()

def load_datasets(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    return image_datasets

def train_model(model, criterion, optimizer, image_datasets, device, epochs=5):
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
                   for x in ['train', 'valid', 'test']}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print_every = 100
    step = 0
    model.to(device)
    dataset_len = len(dataloaders['train'].batch_sampler)
    val_images = len(dataloaders['valid'].batch_sampler) * dataloaders['valid'].batch_size
    print(f'Using the {device} device to train.')
    for e in range(epochs):
        running_loss = 0
        total = 0
        check = 0
        correct = 0
        print(f'\nEpoch {e+1} of {epochs}\n----------------------------')
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Keep a running total of loss for this epoch
            step += 1
            if step % print_every == 0:
                avg_loss = running_loss/step
                accuracy = (correct/total) * 100
                print(f'avg. loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%.')
                check = (ii + 1)
        # Validation
        v_correct = 0
        v_total = 0
        # Disabling gradient calculation
        with torch.no_grad():
            for ii, (inputs, labels) in enumerate(dataloaders['valid']):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()
            correct_perc = 0
            if v_correct > 0:
                pct_correct = (100 * v_correct // v_total)
            print(f'\nValidation accuracy for epoch {e+1} is {pct_correct:d}%.')

    # deep copy the model
    if pct_correct > best_acc:
        best_acc = pct_correct
        best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    print('Training is done.')
    return model

def main():
    args = parse_args()
    image_datasets = load_datasets(args.data_dir)
    device = torch.device("cpu")
    if args.gpu:
        device = torch.device("cuda:0")
    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    if args.arch == "vgg19":
        input_size = 25088
    else:
        input_size = 9216
    classifier = torch.nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', torch.nn.Dropout(p=0.2)),
                          ('fc2', torch.nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    model = model.to(device)
    model = train_model(model, criterion, optimizer, image_datasets, device, epochs=args.epochs)
    model.class_to_idx = image_datasets['train'].class_to_idx
    save_dir = args.save_dir or ""
    checkpoint = {"arch": args.arch, 
                  "classifier": classifier,
                  "state_dict": model.state_dict(),
                  "class_to_idx": model.class_to_idx,
                  }
    
    torch.save(checkpoint, save_dir + "vgg19_p2.pth")
     
if __name__ == "__main__":
    main()