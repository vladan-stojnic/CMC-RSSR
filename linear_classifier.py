import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import wandb
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from dataset import FeatureClassificationDataset
from models.linear_model import LinearClassification
from util import parse_option


def train(epoch, model, criterion, optimizer, train_loader, device, scheduler):
    epoch_loss = 0
    model.train()
    for idx, batch in enumerate(train_loader):
        inputs, targets = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    scheduler.step()
    epoch_loss /= len(train_loader)
    #wandb.log({'train_loss': epoch_loss}, step = epoch)

def test(epoch, model, criterion, test_loader, device, task_type):
    model.eval()
    epoch_bce = 0
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)

            outputs = model(inputs)
            bce = criterion(outputs, targets)
            epoch_bce += bce
            if task_type == 'single-label':
                predictions.append(F.softmax(outputs).cpu().numpy().argmax(axis=1))
            elif task_type == 'multi-label':
                predictions.append(F.sigmoid(outputs).cpu().numpy())
            else:
                raise ValueError('Wrog task type!')
            ground_truth.append(targets.cpu().numpy())

    epoch_bce /= len(test_loader)
    predictions = np.concatenate(predictions, axis = 0)
    ground_truth = np.concatenate(ground_truth, axis = 0)
    
    if task_type == 'single-label':
        metric = accuracy_score(ground_truth, predictions)
    elif task_type == 'multi-label':
        metric = average_precision_score(ground_truth, predictions, average='macro')
    
    print(metric)
    #wandb.log({'test_loss': epoch_bce, 'metric': metric}, step = epoch)
    return metric

args = parse_option()

#wandb.init(config=args, project=args.wandb_project)

scaler = StandardScaler()

train_data = np.load(args.train_data_path)['arr_0']
train_labels = np.load(args.train_data_path)['arr_1']
print(train_labels.shape)
train_data = scaler.fit_transform(train_data)

val_data = np.load(args.val_data_path)['arr_0']
val_labels = np.load(args.val_data_path)['arr_1']
val_data = scaler.transform(val_data)

train_dataset = FeatureClassificationDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataset = FeatureClassificationDataset(val_data, val_labels)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

if train_labels.ndim == 1:
    task_type = 'single-label'
    num_classes = train_labels.max() + 1
    criterion = nn.CrossEntropyLoss()
else:
    task_type = 'multi-label'
    num_classes = train_labels.shape[1]
    criterion = nn.BCEWithLogitsLoss()

model = LinearClassification(input_features = train_data.shape[-1], num_classes = num_classes)

if args.opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
else:
    raise ValueError('Not implemented optimizer! Only Adam supported!')

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, gamma=args.lr_decay_rate)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model.to(device)

#wandb.watch(model)

if not args.evaluate:
    for epoch in range(args.epochs):
        print(epoch)
        train(epoch, model, criterion, optimizer, train_loader, device, scheduler)
        val_metric = test(epoch, model, criterion, val_loader, device, task_type)
    
    torch.save(model, args.resume)
else:
    model = torch.load(args.resume)
    val_metric = test(0, model, criterion, val_loader, device, task_type)
    print(val_metric)
