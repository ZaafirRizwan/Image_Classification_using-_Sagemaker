#Dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

#TODO: Import dependencies for Debugging andd Profiling


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(model, test_loader,criterion):
    '''
    This function takes two arguments and returns None
    
    Parameters:
        -model: Trained Image Classification Network
        -test_loader: DataLoader for test dataset
        
    Returns:
        Trained Image Classification Model
    '''
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output,target,reduction="sum").item() # sum up batch loss
            pred = output.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, epoch):
    '''
    This function takes five arguments and returns Model
    
    Parameters:
        -model: Untrained Image Classification Network
        -train_loader: DataLoader for train dataset
        -criterion: Loss Function
        -optimizer: The optimization algorithm to use
        -epoch: Epoch Number
        
    Returns:
        Trained Image Classification Model
    '''
    
    for batch_idx, (data, target) in enumerate(train_loader):
        model_ft = model_ft.to(device)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    return model
    
def net():
    '''
    This function takes zero parameters and returns a Network
    
    Parameters:
        None
        
    Returns:
        Untrained Image Classification Model
        
    '''
    pretrained_model = models.InceptionResNetV2(include_top=False,weights='imagenet',pooling='avg')
    
    
#     Freezing Pretrained Weights
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
#     Append Fully_Connected layer
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)

    model_ft = model_ft.to(device)
    
    return model
    


def create_data_loaders(data, batch_size):
    '''
    This function takes two arguments and returns Dataloader

    Parameters:
        -data: dataset of train and test images
        -batch_size: No of Images feed into the network at a time

    Returns:
        Dataloader i.e Train and test
    '''
    
    train_dataset_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataset_loader  = torch.utils.data.DataLoader(data , batch_size=batch_size, shuffle=False,num_workers=1)
    dataloaders = {'train': train_dataset_loader, 'test': test_dataset_loader}
    
    return dataloaders
    
def main(args):
    '''
    Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    
    args=parser.parse_args()
    
    main(args)
