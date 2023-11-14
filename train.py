import argparse
import torch
import json
from torch import nn, optim
import torch.utils.data
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time

def arg_parser():

    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('data_dir', action='store', default='./flowers/')
    parser.add_argument('--arch', dest='arch', default='dense121')
    parser.add_argument('--save_dir', dest='save_dir', default='checkpoint.pth')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float)
    parser.add_argument('--hidden_units1', dest='hidden_units1', type=int)
    parser.add_argument('--hidden_units2', dest='hidden_units2', type=int)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('dropout', dest='dropout', type=float)
    parser.add_argument('--gpu', dest='gpu', action='store', default='gpu')

    args = parser.parse_args()
    return args
    # data_dir = args.data_dir
    # gpu = args.gpu
    # epochs = args.epochs
    # arch = args.arch
    # learning_rate = args.learning_rate
    # hidden_units = args.hidden_units
    # dropout = args.dropout
    # save_dir = args.save_dir

# Load the data
def data_loader(data_dir = './flowers'):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_datasets = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloaders = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloaders, validloaders, testloaders, image_datasets

def initial_model(arch='densenet121'):
    # initial model
    model = getattr(models, arch)(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def classifier(model, fc1=512, fc2=256, dropout=0.05):
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, fc1)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(dropout)),
                                       ('fc2', nn.Linear(fc1, fc2)),
                                       ('relu2', nn.ReLU()),
                                       ('fc3', nn.Linear(fc2, 102)),
                                       ('output', nn.LogSoftmax(dim=1))
                                       ]))
    model.classifier = classifier
    return classifier

def train_model(model, trainloaders, validloaders, criterion, optimizer, epochs=5, print_every=20, device='gpu'):

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = print_every
    print('Training started')

    for e in range(epochs):
        # Turn to train mode
        model.train()
        for images, labels in trainloaders:
            steps += 1
            # Move images and labels tensors to the GPU
            images, labels = images.to(device), labels.to(device)

            logps = model.forward(images)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloaders:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)

                        # calculate valid_accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_loss += batch_loss.item()
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {e+1}/{epochs}.. "
                    f"Train Loss: {running_loss/print_every:.3f}.. "
                    f"Valid Loss: {valid_loss/len(validloaders):.3f}.. "
                    f"Valid Accuracy: {valid_accuracy/len(validloaders):.3f}.. ")
                
                # reset running loss to 0
                running_loss = 0
                model.train()
    print('Training completed')
    return model

def test_model(model, testloaders, criterion, device='gpu'):
    test_accuracy = 0

    for images, labels in testloaders:
        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        test_loss = criterion(logps, labels)

        # calculate valid_accuracy
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_loss += test_loss.item()
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test Loss: {test_loss/len(testloaders):.3f}.. ")    
         
def gpu_check():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    return device

def main():

    args = arg_parser()
    
    trainloaders, validloaders, testloaders, image_dataset = data_loader(args.data_dir)
    
    model = initial_model(args.arch)
    # Define hyperparameters
    classifier(model, args.hidden_units1, args.hidden_units2, args.dropout)

    device = gpu_check(args.gpu)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    model = train_model(model, trainloaders, validloaders, args.epochs, criterion, optimizer, device)

    # Test the model
    test_model(model, testloaders, criterion, device)
    
    # Save the checkpoint
    model.class_to_idx = image_dataset['train'].class_to_idx
    model.cpu()
    torch.save({'structure': args.arch,
           'fc1': args.hidden_units1,
           'dropout': args.dropout,
           'fc2': args.hidden_units2,
           'epoch': args.epochs,
           'state_dict': model.state_dict(),
           'class_to_idx': model.class_to_idx,
           'optimizer_dict': optimizer.state_dict()}, args.save_dir)
    print('Checkpoint saved')

if __name__ == '__main__':
    main()









