#Dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import boto3

import argparse

# For Profiling
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

#  For Debugging
import smdebug.pytorch as smd


s3 = boto3.client('s3')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(model, test_loader,criterion,hook):
    '''
    This function takes two arguments and returns None
    
    Parameters:
        -model: Trained Image Classification Network
        -test_loader: DataLoader for test dataset
        -hook: hook for saving model tensors during testing for ananlyzing model behavior
        
    Returns:
        None
    '''
    
    # Setting SMDEBUG hook for testing Phase
    hook.set_mode(smd.modes.EVAL)
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

def train(model, train_loader, criterion, optimizer, epoch,hook):
    '''
    This function takes five arguments and returns None
    
    Parameters:
        -model: Untrained Image Classification Network
        -train_loader: DataLoader for train dataset
        -criterion: Loss Function
        -optimizer: The optimization algorithm to use
        -epoch: Epoch Number
        -hook: hook for saving model tensors during training for ananlyzing model behavior
        
    Returns:
        None
    '''
    # Setting SMDEBUG hook for model training loop
    hook.set_mode(smd.modes.TRAIN)
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

    
def net(args):
    '''
    This function takes one parameters and returns a Network
    
    Parameters:
        args: Command Line Arguments
        
    Returns:
        Untrained Image Classification Model
        
    '''
    pretrained_model = models.__dict__[args.model](pretrained=True))
    
    
    # Freezing Pretrained Weights
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    # Append Fully_Connected layer
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, 10)

    model_ft = pretrained_model.to(device)
    
    return pretrained_model
    


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

    # Initializing Model
    model = net(args)
    
    ##################Debugging###################
    # Registering SMDEBUG hook to save output tensors.
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    ##############################################
    
    # Creating Loss Function and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    
    # Read Dataset
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.ToTensor(), normalize]
    )
    
    create_data_loaders(data,args.batch_size)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    for epoch in range(len(args.epochs)):
        train(model, train_loader, loss_criterion, optimizer, epoch, hook)
        test(model, test_loader, criterion, hook)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, "classificationmodel.pt")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Deep Learning on Amazon Sagemaker")
    parser.add_argument("--batch-size",
                        type=int,
                        default=64,
                        metavar="N",
                        help="input batch size for training (default: 64)",
                       )
    
    parser.add_argument("--epochs",
                       type=int,
                       default=3,
                       metavar="N",
                       help="input batch size for training (default: 64)"
                       )
    parser.add_argument("--lr",
                   type=float,
                   default=1.0,
                   metavar="LR",
                   help="learning rate (default: 1.0)",
                   )
    parser.add_argument("--model", 
                        type=str, 
                        default="InceptionResNetV2")
    
    args = parser.parse_args()
    
    #  Printing Arguments
    for key, value in vars(opt).items():
        print(f"{key}:{value}")
    
    
    main(args)
