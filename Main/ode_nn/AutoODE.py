import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Auto_ODE_SEIR(nn.Module):
    def __init__(self, solver = "Euler"):
        super(Auto_ODE_SEIR, self).__init__()
        self.initial = torch.nn.Parameter(torch.rand(4)/4) #10000.0)#torch.tensor([0.99, 0., 0.01, 0.]).cuda()#
        self.beta = torch.nn.Parameter(torch.rand(1)) #torch.tensor(0.9).cuda()#
        self.gamma = torch.nn.Parameter(torch.rand(1)) #torch.tensor(0.1).cuda()#
        self.sigma = torch.nn.Parameter(torch.rand(1)) #torch.tensor(0.5).cuda()#
        self.step = torch.tensor(0.5)
        self.solver = solver
        
    def Euler(self, t):
        S_pred = [self.initial[0].reshape(-1,1)]
        E_pred = [self.initial[1].reshape(-1,1)]
        I_pred = [self.initial[2].reshape(-1,1)]
        R_pred = [self.initial[3].reshape(-1,1)]
        for n in range(len(t)-1):
            S_pred.append((S_pred[n] - self.beta*S_pred[n]*I_pred[n]*self.step).reshape(-1,1))
            E_pred.append((E_pred[n] + (self.beta*S_pred[n]*I_pred[n] - self.sigma*E_pred[n])*self.step).reshape(-1,1))
            I_pred.append((I_pred[n] + (self.sigma*E_pred[n] - self.gamma*I_pred[n])*self.step).reshape(-1,1))
            R_pred.append((R_pred[n] + self.gamma*I_pred[n]*self.step).reshape(-1,1))
        y_pred = torch.cat([torch.cat(S_pred, dim = 0),
                            torch.cat(E_pred, dim = 0),
                            torch.cat(I_pred, dim = 0), 
                            torch.cat(R_pred, dim = 0)], dim = 1)
        return y_pred
    
    def RK4(self, t):
        S_pred = [self.initial[0].reshape(-1,1)]
        E_pred = [self.initial[1].reshape(-1,1)]
        I_pred = [self.initial[2].reshape(-1,1)]
        R_pred = [self.initial[3].reshape(-1,1)]
        
        for n in range(len(t)-1):
            k1 = self.beta*S_pred[n]*I_pred[n]*self.step #dt * f(t[n], y[n]) 
            k2 = (self.beta*(S_pred[n]+ k1/2)*I_pred[n])*self.step #dt * f(t[n] + dt/2, y[n] + k1/2)
            k3 = (self.beta*(S_pred[n]+ k2/2)*I_pred[n])*self.step #dt * f(t[n] + dt/2, y[n] + k2/2)
            k4 = (self.beta*(S_pred[n]+ k3)*I_pred[n])*self.step #dt * f(t[n] + dt, y[n] + k3)            
            S_pred.append((S_pred[n] - 1/6 * (k1 + 2*k2 + 2*k3 + k4)).reshape(-1,1))
            
            k1 = (self.beta*S_pred[n]*I_pred[n] - self.sigma*E_pred[n])*self.step #dt * f(t[n], y[n]) 
            k2 = (self.beta*S_pred[n]*I_pred[n] - self.sigma*(E_pred[n] +  k1/2))*self.step #dt * f(t[n] + dt/2, y[n] + k1/2)
            k3 = (self.beta*S_pred[n]*I_pred[n] - self.sigma*(E_pred[n] +  k2/2))*self.step #dt * f(t[n] + dt/2, y[n] + k2/2)
            k4 = (self.beta*S_pred[n]*I_pred[n] - self.sigma*(E_pred[n] +  k3))*self.step #dt * f(t[n] + dt, y[n] + k3)            
            E_pred.append((E_pred[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)).reshape(-1,1))
            
            k1 = (self.sigma*E_pred[n] - self.gamma*I_pred[n])*self.step #dt * f(t[n], y[n]) 
            k2 = (self.sigma*E_pred[n] - self.gamma*(I_pred[n] + k1/2))*self.step #dt * f(t[n] + dt/2, y[n] + k1/2)
            k3 = (self.sigma*E_pred[n] - self.gamma*(I_pred[n] + k2/2))*self.step #dt * f(t[n] + dt/2, y[n] + k2/2)
            k4 = (self.sigma*E_pred[n] - self.gamma*(I_pred[n] + k3))*self.step #dt * f(t[n] + dt, y[n] + k3)   
            I_pred.append((I_pred[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)).reshape(-1,1))
            
            R_pred.append((R_pred[n] + self.gamma*I_pred[n]*self.step).reshape(-1,1))

        y_pred = torch.cat([torch.cat(S_pred, dim = 0), 
                            torch.cat(E_pred, dim = 0),
                            torch.cat(I_pred, dim = 0), 
                            torch.cat(R_pred, dim = 0)], dim = 1)
        return y_pred
            
            
    def forward(self, t):
        if self.solver == "Euler":
            return self.Euler(t)
        elif self.solver == "RK4":
            return self.RK4(t)
        else:
            print("Error")        
            
            
def weight_fun(num_steps, function = "linear", feat_weight = False):    
    if function == "linear":
        weight = torch.linspace(0, 1, num_steps).reshape(1,-1,1)*2/num_steps
    if function == "sqrt":
        sqrt_func = lambda x: torch.sqrt(x)
        weight = sqrt_func(torch.linspace(0, 1, num_steps).reshape(1,-1,1))/torch.sum(sqrt_func(torch.linspace(0, 1, num_steps)))
    return weight

class Auto_ODE_LV(nn.Module):
    def __init__(self, num_time_series, p0):
        super(Auto_ODE_LV, self).__init__()
        self.num_time_series = num_time_series
        self.p0 = p0#nn.Parameter(torch.rand(num_time_series).float())
        self.r = nn.Parameter(torch.rand(num_time_series).float()/10)
        self.k = nn.Parameter(torch.rand(num_time_series).float()*100)
        self.A = nn.Parameter(torch.rand(num_time_series, num_time_series).float()/10)
        
    def solve(self, num_steps):
        p = [] 
        p.append(self.p0)
        for n in range(num_steps-1): # element-wise vector division and multiplication
            mat_vec_prod = torch.mm(self.A, p[n].reshape(-1, 1)).squeeze(-1)
            p.append((1 + self.r * (1 - mat_vec_prod )) * p[n])#/ self.k

        return torch.cat(p, dim=0).reshape(num_steps, self.num_time_series)#.T

    def forward(self, num_steps):
        return self.solve(num_steps)       
    
# preds = []
# trues = []
# for xx, yy in test_down_loader:
#     time = torch.tensor(np.linspace(0, 60, 61), requires_grad=True).float()[:60]#.cuda()
#     model_ode = #Auto_ODE_SEIR(solver = "Euler")#.cuda()
#     y_exact = xx[0]

#     optimizer = torch.optim.Adam(model_ode.parameters(), 0.01)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 100, gamma=0.95)
#     num_epochs = 5000
#     loss_fun = torch.nn.MSELoss()
#     tqdm_epochs = tqdm(range(num_epochs))
#     min_loss = 10

#     for e in tqdm_epochs:
#         scheduler.step()
#         y_approx = model_ode(time[:30])
#         loss = loss_fun(y_approx, y_exact[:30])
#         #loss_weight = weight_fun(20, function = "linear", feat_weight = True)
#         #loss = torch.mean(loss_weight*loss_fun(y_approx, y_exact[:30])) 

#         if loss.item() < min_loss:
#             best_model_ode = model_ode
#             min_loss = torch.sum(loss).item()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         tqdm_epochs.set_postfix({'loss': torch.sum(loss).item()})
        
#     y_pred = best_model_ode(time).cpu().data.numpy()
#     y_exact = yy[0].cpu().data.numpy()
#     preds.append(y_pred)
#     trues.append(y_exact)
#     if len(preds) == 10:
#         break
# y_preds = np.array(preds)
# y_trues = np.array(trues)
# print(np.sum(y_preds!=y_preds))
# np.sqrt(np.mean((np.diff(y_preds, axis = 1) - np.diff(y_trues, axis = 1))[:,-40:]**2))*1000

# y_preds = np.array(preds)
# y_trues = np.array(trues)
# torch.save({"preds": y_preds,
#             "trues": y_trues,
#             "rmse": np.sqrt(np.mean((np.diff(y_preds, axis = 1) - np.diff(y_trues, axis = 1))[:,-40:]**2))*1000},
#             "AutoODE_SEIR_init_extra.pt")