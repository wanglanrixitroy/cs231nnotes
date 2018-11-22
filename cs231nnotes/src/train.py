from data import DogbreedDataset
from preprocessor import Rescale,  ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision import transforms, models
from splitdata import splitdata
import torch.nn as nn
from torch import optim
import torch
from torch.autograd import Variable
import time
import numpy as np

def get_data():
    # load data
    dogdataset = DogbreedDataset(csv_file='all/labels.csv', root_dir='all/train', transform=transforms.Compose([Rescale((224,224)),ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))

    # print (dogdataset[100])
    batch_size = 64

    # split data
    train_sampler, valid_sampler, test_sampler = splitdata(dogdataset, 0.7, 0.25, 50)
    train_loader = DataLoader(dogdataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=2)
    validation_loader = DataLoader(dogdataset, batch_size=batch_size, sampler=valid_sampler, shuffle=False)
    test_loader = DataLoader(dogdataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)

    dataset_sizes = {'train':len(train_sampler), 'valid':len(valid_sampler), 'test':len(test_sampler)}

    dataloaders = {'train':train_loader, 'valid':validation_loader, 'test':test_loader}
    return dataloaders

def myMultiLabelCrossEntropyLoss(output, labels):
    

def get_optimizer(param, learning_rate, optim_name='SGD', weight_decay=5e-2, nesterov=True):
    if optim_name == 'Adam':
        optimizer = optim.Adam(param, lr=learning_rate, weight_decay=weight_decay)

    elif optim_name == 'RMSprop':
        optimizer = optim.RMSprop(param, lr=learning_rate, weight_decay=weight_decay, alpha=0.9, eps=1.0, momentum=0.9)

    else:
        optimizer = optim.SGD(param, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)

    return optimizer

def train(num_epoch):
    # Create a pretrain model
    resmodel = models.resnet34(pretrained=True)
    num_ftrs = resmodel.fc.in_features
    resmodel.fc = nn.Linear(num_ftrs, 120)

    # get GPU
    if torch.cuda.is_available():
        resmodel = resmodel.cuda()
        is_cuda = True
    # define optimizer and learning rate scheduler
    learning_rate = 0.001
    optimizer = get_optimizer(resmodel.parameters(), learning_rate=learning_rate, optim_name='SGD')
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # define loss criterion
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCEWithLogitsLoss()

    best_model_wts = resmodel.state_dict()
    best_acc = 0.0

    start = time.time()

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)
        losses = []

        # each epcoh has a training, validation and test phase
        for phase in ['train', 'valid', 'test']:
            if phase == 'train':
                lr_scheduler.step()
                resmodel.train(True)
            else:
                resmodel.train(False)

            running_loss = 0.
            running_corrests = 0.

            # Iterate over data
            dataloaders = get_data()
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data['image'], data['landmarks']

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = resmodel(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                losses.append(loss.data.mean())

            epoch_loss = np.mean(losses)
            epoch_acc = 1 - np.mean(losses)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = resmodel.state_dict()

        print()
    time_spent = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_spent // 60, time_spent % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    resmodel.load_state_dict(best_model_wts)
    return resmodel






if __name__ == '__main__':
    train(num_epoch=5)
