import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from sklearn.metrics import classification_report

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_step(model, optimizer, loader, loss):
    
    model.train()
    
    loss_hist = []
    acc_hist = []
    
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        yh = model(x)
        optimizer.zero_grad()
        loss_ = loss(yh, y)
        loss_.backward()
        optimizer.step()
        
        loss_hist.append(loss_.item())
        acc_hist.append((yh.argmax(dim=1)==y).sum().item())
        
    return np.array(loss_hist).mean(), np.array(acc_hist).mean()

def valid_step(model, loader, loss):
    
    model.eval()
    
    loss_hist = []
    acc_hist = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            yh = model(x)
            loss_ = loss(yh, y)

            loss_hist.append(loss_.item())
            acc_hist.append((yh.argmax(dim=1)==y).sum().item())
        
    return np.array(loss_hist).mean(), np.array(acc_hist).mean()

def train(model, valid_dataset, train_dataset, bs, epochs,decay = [100, 150], lr = 1e-1, path = 'best_model.ckpt'):
    
    model = model.to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=decay)
    
    loss = torch.nn.CrossEntropyLoss()
    
    val_loss_hist = []
    train_loss_hist = []
    
    val_acc_hist = []
    train_acc_hist = []
    
    best_model = deepcopy(model)
    partience = 0
    
    train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True, pin_memory = True)
    valid_loader = DataLoader(valid_dataset, batch_size = bs, shuffle = False, pin_memory = True)
        
    pbar = tqdm(total=epochs)
    pbar.set_description('training process')
    
    for _ in range(epochs):
        
        train_loss, train_acc = train_step(model, optimizer, train_loader, loss)
        valid_loss, valid_acc = valid_step(model, valid_loader, loss)
        
        scheduler.step()
        
        val_loss_hist.append(valid_loss.item())
        train_loss_hist.append(train_loss.item())
        val_acc_hist.append(valid_acc.item()/bs)
        train_acc_hist.append(train_acc.item()/bs)
        
        if valid_acc.item()/bs == max(val_acc_hist):
            best_model = deepcopy(model)
            partience = 0
            
        if partience>=10:
            print('early stopping')
            break
        
        pbar.update(1)
        pbar.set_description('val loss: {:.3f}, val acc: {:.3f}, train loss: {:.3f}, train acc: {:.3f}'.format(
            valid_loss, valid_acc/bs, train_loss, train_acc/bs))
        
    pbar.close()
    print('training finished with best accuracy : {:.3f}'.format(max(val_acc_hist)))
    torch.save(best_model.state_dict(), path)
    
    
def validate(model, valid_dataset):
    
    valid_loader = DataLoader(valid_dataset, batch_size = 128, shuffle = False, pin_memory = True)
    
    pbar = tqdm(total=len(valid_loader))
    pbar.set_description('valid model')
    
    model.to(DEVICE)
    model.eval()
    
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            yh = model(x)

            y_true.append(y)
            y_pred.append(yh.argmax(dim=1))
            
            pbar.update(1)
    
    pbar.close()
            
    y_pred = torch.cat(y_pred, dim = 0).cpu().numpy()
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    
    print(classification_report(y_true, y_pred))