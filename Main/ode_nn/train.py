import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class Dataset(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, direc, entire_target = False, N = 1.0):
        self.mid = mid
        self.input_length = input_length
        self.output_length = output_length
        self.direc = direc
        self.list_IDs = indices
        self.entire_target = entire_target
        self.N = N
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        sample = torch.load(self.direc + str(ID) + ".pt")
        x = sample[(self.mid-self.input_length):self.mid]/self.N
        if self.entire_target:
            y = sample[(self.mid-self.input_length):(self.mid+self.output_length)]/self.N
        else:
            y = sample[self.mid:(self.mid+self.output_length)]/self.N
        return x.float(), y.float()
    
    
def train_epoch(model, data_loader, optimizer, loss_fun, feed_tgt = False):
    preds = []
    trues = []
    mse = []
    for xx, yy in data_loader:
        xx, yy = xx.to(device), yy.to(device)
        loss = 0
        if feed_tgt:
            
            yy_pred = model(xx, yy.shape[1], yy)
        else:
            yy_pred = model(xx, yy.shape[1])
        loss = loss_fun(yy_pred, yy)
        mse.append(loss.item())
        trues.append(yy.cpu().data.numpy())
        preds.append(yy_pred.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    preds = np.concatenate(preds, axis = 0)
    trues = np.concatenate(trues, axis = 0)
    return preds, trues, np.round(np.sqrt(np.mean(mse)), 5)

def eval_epoch(model, data_loader, loss_fun, concat_input = False):
    preds = []
    trues = []
    mse = []
    with torch.no_grad():
        for xx, yy in data_loader:
            xx, yy = xx.to(device), yy.to(device)
            loss = 0
            yy_pred = model(xx, yy.shape[1])
            loss = loss_fun(yy_pred, yy)
            mse.append(loss.item())
            if concat_input:
                trues.append(torch.cat([xx, yy], dim = 1).cpu().data.numpy())
            else:
                trues.append(yy.cpu().data.numpy())
            preds.append(yy_pred.cpu().data.numpy())

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)

    return preds, trues, np.round(np.sqrt(np.mean(mse)), 5)
    
class Dataset_graph(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, direc, entire_target = False, N = 1.0, stack = True):
        self.mid = mid
        self.input_length = input_length
        self.output_length = output_length
        self.direc = direc
        self.list_IDs = indices
        self.entire_target = entire_target
        self.N = N
        self.stack = stack
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        sample = torch.load(self.direc + str(ID) + ".pt")
        x = sample[:,(self.mid-self.input_length):self.mid]/self.N
        if self.entire_target:
            y = sample[:,(self.mid-self.input_length):(self.mid+self.output_length)]/self.N
        else:
            y = sample[:, self.mid:(self.mid+self.output_length)]/self.N
        if self.stack:
            return x.reshape(x.shape[0], -1).float(), y.float()
        return x.float(), y.float()
    
def train_epoch_graph(model, data_loader, optimizer, loss_fun, graph):
    preds = []
    trues = []
    mse = []
    for xx, yy in data_loader:
        xx, yy = xx.to(device), yy.to(device)
        loss = 0
        yy_pred = model(graph, xx, yy.shape[2])
        loss = loss_fun(yy_pred, yy)
        mse.append(loss.item())
        trues.append(yy.cpu().data.numpy())
        preds.append(yy_pred.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    preds = np.concatenate(preds, axis = 0)
    trues = np.concatenate(trues, axis = 0)
    return preds, trues, np.round(np.sqrt(np.mean(mse)), 5)

def eval_epoch_graph(model, data_loader, loss_fun, graph):
    preds = []
    trues = []
    mse = []
    with torch.no_grad():
        for xx, yy in data_loader:
            xx, yy = xx.to(device), yy.to(device)
            loss = 0
            yy_pred = model(graph, xx, yy.shape[2])
            loss = loss_fun(yy_pred, yy)
            mse.append(loss.item())
            if yy.shape[1] != 60:
                trues.append(torch.cat([xx.reshape(xx.shape[0], xx.shape[1], -1, 3), yy], dim = 2).cpu().data.numpy())
            else:
                trues.append(yy.cpu().data.numpy())
            preds.append(yy_pred.cpu().data.numpy())

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)

    return preds, trues, np.round(np.sqrt(np.mean(mse)), 5)