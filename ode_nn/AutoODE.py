import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_fun(num_steps, function = "linear", feat_weight = False):    
    if function == "linear":
        weight = torch.linspace(0, 1, num_steps).reshape(1,-1,1)*2 / num_steps
    if function == "sqrt":
        sqrt_func = lambda x: torch.sqrt(x)
        weight = sqrt_func(torch.linspace(0, 1, num_steps).reshape(1,-1,1)) / torch.sum(sqrt_func(torch.linspace(0, 1, num_steps)))
    if feat_weight:
        weight = weight.repeat(1,1,3)
        weight[:,:,2] =  weight[:,:,2] #* 12
        weight[:,:,1] =  weight[:,:,1] #* 4
    return weight

class PiecewiseLinearModel(nn.Module):
    def __init__(self, n_breaks, num_regions):
        super(PiecewiseLinearModel, self).__init__()
        self.breaks = nn.Parameter(torch.rand((num_regions, 1, n_breaks)))
        self.linear = nn.Linear(n_breaks + 1, 1) 
    def forward(self, xx):
        if len(xx.shape) < 3:
            xx = xx.unsqueeze(-1)
        out = torch.cat([xx, F.relu(xx - self.breaks)],2)
        return self.linear(out).squeeze(-1)
    
class AutoODE_COVID(nn.Module):
    def __init__(self, initial_I, initial_R, initial_D, num_regions, 
                 solver = "Euler", Corr = False, low_rank = None, 
                 n_breaks = 0, graph = None, beta = None):
        super(AutoODE_COVID, self).__init__()
        self.num_regions = num_regions
        self.init_I = initial_I
        self.init_R = initial_R
        self.init_D = initial_D        
        self.init_E = nn.Parameter(torch.tensor([0.5] * num_regions).float().to(device)) 
        self.init_S = nn.Parameter(torch.tensor([0.5] * num_regions).float().to(device)) 
        
        if Corr:
            if num_regions == 1:
                self.A = torch.ones(1, 1).to(device)
            else:
                if low_rank:
                    if symmetric: 
                        self.B = nn.Parameter(torch.rand(num_regions, low_rank).to(device))
                        self.A = torch.mm(self.B, self.B.T)
                    else:
                        self.B = nn.Parameter(torch.rand(num_regions, low_rank).to(device))
                        self.C = nn.Parameter(torch.rand(low_rank, num_regions).to(device))
                        self.A = torch.mm(self.B, self.C)
                else:
                    self.A = nn.Parameter(torch.rand(num_regions, num_regions).to(device))
        else:
            self.A = np.zeros((num_regions, num_regions))
            np.fill_diagonal(self.A, 1.0)
            self.A = torch.from_numpy(self.A).float().to(device)

        self.graph = graph
        if beta is None:
            if n_breaks > 0:
                self.plm = PiecewiseLinearModel(n_breaks = n_breaks, num_regions = num_regions)
            else:
                self.beta = nn.Parameter(torch.rand(num_regions).to(device)/10)
        else:
            self.beta = beta
        self.n_breaks = n_breaks
        self.gamma = nn.Parameter(torch.rand(num_regions).to(device)/10)
        self.sigma = nn.Parameter(torch.rand(num_regions).to(device)/10)
        self.mu = nn.Parameter(torch.rand(num_regions).to(device)/10)
        self.step = torch.tensor(0.01).float().to(device)
        self.a = nn.Parameter(torch.rand(num_regions).to(device)/10)
        self.b = nn.Parameter(torch.rand(num_regions).to(device)/10)
        self.solver = solver
        self.init_U = (1-self.mu)*self.sigma*self.init_E
        
    def Euler(self, num_steps):
        t = torch.linspace(1, num_steps, num_steps).repeat(self.num_regions, 1)
        if self.n_breaks > 0:
            beta = self.plm(t)
        else:
            beta = self.beta.repeat(1, num_steps)
        S_pred = [self.init_S]
        E_pred = [self.init_E]
        I_pred = [self.init_I]
        R_pred = [self.init_R]
        D_pred = [self.init_D]
        for n in range(num_steps - 1):
            if self.graph is None:
                S_pred.append(S_pred[n] - beta[:, n+1] * (torch.mm(self.A, ((I_pred[n] + E_pred[n]) * S_pred[n]).reshape(-1,1)).squeeze(1)) * self.step)
                E_pred.append(E_pred[n] + (beta[:, n+1] * S_pred[n] * (I_pred[n]+ E_pred[n]) - self.sigma * E_pred[n]) * self.step)
            else:
                S_pred.append(S_pred[n] - beta[:, n+1] * (torch.mm(self.graph*self.A, ((I_pred[n] + E_pred[n]) * S_pred[n]).reshape(-1,1)).squeeze(1)) * self.step)
                E_pred.append(E_pred[n] + (beta[:, n+1] * (torch.mm(self.graph*self.A, ((I_pred[n] + E_pred[n]) * S_pred[n]).reshape(-1,1)).squeeze(1)) - self.sigma * E_pred[n]) * self.step)
             
            I_pred.append(I_pred[n] + (self.mu * self.sigma * E_pred[n] - self.gamma*I_pred[n]) * self.step)
            R_pred.append(R_pred[n] + self.gamma * I_pred[n] * self.step)
            D_pred.append(D_pred[n] + self.a * torch.exp(- self.b * (n + 1) * self.step) * (R_pred[n+1] - R_pred[n]))
        y_pred = torch.cat([torch.stack(S_pred).transpose(0,1).unsqueeze(-1),
                           (torch.stack(E_pred)*(1-self.mu.unsqueeze(0))*self.sigma.unsqueeze(0)).transpose(0,1).unsqueeze(-1),
                            torch.stack(E_pred).transpose(0,1).unsqueeze(-1),
                            torch.stack(I_pred).transpose(0,1).unsqueeze(-1),
                            torch.stack(R_pred).transpose(0,1).unsqueeze(-1),
                            torch.stack(D_pred).transpose(0,1).unsqueeze(-1)], dim = -1)
        return y_pred
    
    def f_S(self, S_n, I_n, E_n, beta, n): 
        return -beta[:, n+1] * (torch.mm(self.A, ((I_n + E_n) * S_n).reshape(-1,1)).squeeze(1))
    
    def f_E(self, S_n, I_n, E_n, beta, n): 
        return beta[:, n+1] * S_n * (I_n + E_n) - self.sigma * E_n
    
    def f_I(self, I_n, E_n):
        return self.mu * self.sigma * E_n - self.gamma*I_n
    
    def f_R(self, I_n):
        return self.gamma*I_n

    def RK4_update(self, f_n, k1, k2, k3, k4):
        return f_n + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4) * self.step
    
    def RK4(self, num_steps):
        
        t = torch.linspace(1, num_steps, num_steps).repeat(self.num_regions, 1)
        if self.n_breaks > 0:
            beta = self.plm(t)
        else:
            beta = self.beta.repeat(1, num_steps)
            
        S_pred = [self.init_S]
        E_pred = [self.init_E]
        I_pred = [self.init_I]
        R_pred = [self.init_R]
        D_pred = [self.init_D]
        for n in range(num_steps-1):
            # dt * f(t[n], y[n]) 
            k1_S = self.f_S(S_pred[n], I_pred[n], E_pred[n], beta, n)
            k1_E = self.f_E(S_pred[n], I_pred[n], E_pred[n], beta, n)
            k1_I = self.f_I(I_pred[n], E_pred[n])
            k1_R = self.f_R(I_pred[n])
            
            # dt * f(t[n] + dt/2, y[n] + k1/2)
            S_plus_k1_half = S_pred[n] + k1_S / 2 * self.step
            I_plus_k1_half = I_pred[n] + k1_I / 2 * self.step
            E_plus_k1_half = E_pred[n] + k1_E / 2 * self.step
            
            k2_S = self.f_S(S_plus_k1_half, I_plus_k1_half, E_plus_k1_half, beta, n)
            k2_E = self.f_E(S_plus_k1_half, I_plus_k1_half, E_plus_k1_half, beta, n)
            k2_I = self.f_I(I_plus_k1_half, E_plus_k1_half)
            k2_R = self.f_R(I_plus_k1_half)
            
            # dt * f(t[n] + dt/2, y[n] + k2/2)
            S_plus_k2_half = S_pred[n] + k2_S / 2 * self.step
            I_plus_k2_half = I_pred[n] + k2_I / 2 * self.step
            E_plus_k2_half = E_pred[n] + k2_E / 2 * self.step
            
            k3_S = self.f_S(S_plus_k2_half, I_plus_k2_half, E_plus_k2_half, beta, n)
            k3_E = self.f_E(S_plus_k2_half, I_plus_k2_half, E_plus_k2_half, beta, n)
            k3_I = self.f_I(I_plus_k2_half, E_plus_k2_half)
            k3_R = self.f_R(I_plus_k2_half)
            
            # dt * f(t[n] + dt, y[n] + k3) 
            S_plus_k3 = S_pred[n] + k3_S * self.step
            I_plus_k3 = I_pred[n] + k3_I * self.step
            E_plus_k3 = E_pred[n] + k3_E * self.step
            
            k4_S = self.f_S(S_plus_k3, I_plus_k3, E_plus_k3, beta, n)           
            k4_E = self.f_E(S_plus_k3, I_plus_k3, E_plus_k3, beta, n)
            k4_I = self.f_I(I_plus_k3, E_plus_k3) 
            k4_R = self.f_R(I_plus_k3)

            S_pred.append(self.RK4_update(S_pred[n], k1_S, k2_S, k3_S, k4_S))
            E_pred.append(self.RK4_update(E_pred[n], k1_E, k2_E, k3_E, k4_E))
            I_pred.append(self.RK4_update(I_pred[n], k1_I, k2_I, k3_I, k4_I))
            R_pred.append(self.RK4_update(R_pred[n], k1_R, k2_R, k3_R, k4_R))
            
        for n in range(num_steps - 1):
            D_pred.append(D_pred[n] + (self.a * (n * self.step) + self.b) * (R_pred[n+1] - R_pred[n]))

        y_pred = torch.cat([torch.stack(S_pred).transpose(0,1).unsqueeze(-1),
                            (torch.stack(E_pred)*(1-self.mu.unsqueeze(0))*self.sigma.unsqueeze(0)).transpose(0,1).unsqueeze(-1),
                            torch.stack(E_pred).transpose(0,1).unsqueeze(-1),
                            torch.stack(I_pred).transpose(0,1).unsqueeze(-1),
                            torch.stack(R_pred).transpose(0,1).unsqueeze(-1),
                            torch.stack(D_pred).transpose(0,1).unsqueeze(-1)], dim = -1)
        return y_pred
    
        
    def forward(self, num_steps):
        if self.solver == "Euler":
            return self.Euler(num_steps)[:,:,-3:]
        elif self.solver == "RK4":
            return self.RK4(num_steps)[:,:,-3:]
        else:
            print("Error")        
            

##############################################################################################################################3
class Auto_ODE_SEIR(nn.Module):
    def __init__(self, initial, solver = "Euler"):
        super(Auto_ODE_SEIR, self).__init__()
        self.initial = torch.nn.Parameter(initial) #10000.0)#torch.tensor([0.99, 0., 0.01, 0.]).cuda()#torch.rand(4)/4
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
        for n in range(t-1):
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
        
        for n in range(t-1):
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
            
            
            
# def weight_fun(num_steps, function = "linear", feat_weight = False):    
#     if function == "linear":
#         weight = torch.linspace(0, 1, num_steps).reshape(1,-1,1)*2/num_steps
#     if function == "sqrt":
#         sqrt_func = lambda x: torch.sqrt(x)
#         weight = sqrt_func(torch.linspace(0, 1, num_steps).reshape(1,-1,1))/torch.sum(sqrt_func(torch.linspace(0, 1, num_steps)))
#     return weight

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
    

    
class Auto_ODE_FHN(nn.Module):
    def __init__(self, initials, solver = "Euler"):
        super(Auto_ODE_FHN, self).__init__()
        self.x0 = initials[0]
        self.y0 = initials[1]
        self.solver = solver
        #print(self.linear(torch.rand(1)).shape)
        
        self.a = nn.Parameter(torch.tensor(0.3).float())#self.embed_a(torch.rand(1).float())#
        self.b = nn.Parameter(torch.tensor(0.3).float())#self.embed_b(torch.rand(1).float())#
        self.c = nn.Parameter(torch.tensor(2.0).float())#self.embed_c(torch.rand(1).float())#n
        self.step = torch.tensor(0.1)#)#torch.tensor(1.0)#nn.Parameter(
        
    def Euler(self, num_steps):
        #self.a, self.b, self.c, self.d = self.embed(torch.rand(1).reshape(1,1))[0]
        x = [self.x0.reshape(-1,1)] 
        y = [self.y0.reshape(-1,1)]
        for n in range(num_steps-1): 
            x.append(x[n] + (self.c*(x[n] + y[n] - x[n]*x[n]*x[n]/3)).reshape(-1,1)*self.step)
            y.append(y[n] - ((1/self.c)*(x[n] + self.b*y[n] - self.a)).reshape(-1,1)*self.step)
       #print((self.c*(x[n] + y[n] - x[n]**3/3)).reshape(-1,1).shape)
        return torch.cat([torch.cat(x, dim = 0), torch.cat(y, dim = 0)], dim = 1)
    
    def f_x(self, x_n, y_n, c): 
        return (c*(x_n + y_n - x_n**3/3)).reshape(-1,1)
    
    def f_y(self, x_n, y_n, a, b, c): 
        return - ((1/c)*(x_n + b*y_n - a)).reshape(-1,1)
    
    def RK4_update(self, f_n, k1, k2, k3, k4):
        return f_n + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4) * self.step
    
    def RK4(self, num_steps):
        #self.a, self.b, self.c, self.d = self.embed(torch.rand(1).reshape(1,1))[0]
        x = [self.x0.reshape(-1,1)] 
        y = [self.y0.reshape(-1,1)]
        
        for n in range(num_steps-1):
            # dt * f(t[n], y[n]) 
            k1_x = self.f_x(x[n], y[n], self.c)
            k1_y = self.f_y(x[n], y[n], self.a, self.b, self.c)

            # dt * f(t[n] + dt/2, y[n] + k1/2)
            x_plus_k1_half = x[n] + k1_x / 2 * self.step
            y_plus_k1_half = y[n] + k1_y / 2 * self.step
            
            k2_x = self.f_x(x_plus_k1_half, y_plus_k1_half, self.c)
            k2_y = self.f_y(x_plus_k1_half, y_plus_k1_half, self.a, self.b, self.c)

            # dt * f(t[n] + dt/2, y[n] + k2/2)
            x_plus_k2_half = x[n] + k2_x / 2 * self.step
            y_plus_k2_half = y[n] + k2_y / 2 * self.step
            
            k3_x = self.f_x(x_plus_k2_half, y_plus_k2_half, self.c)
            k3_y = self.f_y(x_plus_k2_half, y_plus_k2_half, self.a, self.b, self.c)
            
            # dt * f(t[n] + dt, y[n] + k3) 
            x_plus_k3 = x[n] + k3_x * self.step
            y_plus_k3 = y[n] + k3_y * self.step
            
            k4_x = self.f_x(x_plus_k3, y_plus_k3, self.c)    
            k4_y = self.f_y(x_plus_k3, y_plus_k3, self.a, self.b, self.c)
            

            x.append(self.RK4_update(x[n], k1_x, k2_x, k3_x, k4_x))
            y.append(self.RK4_update(y[n], k1_y, k2_y, k3_y, k4_y))
        #print(x[-1], y[-1])
        return torch.cat([torch.cat(x, dim = 0), torch.cat(y, dim = 0)], dim = 1)
    def forward(self, num_steps):
        if self.solver == "Euler":
            return self.Euler(num_steps)    
        elif self.solver == "RK4":
            return self.RK4(num_steps)    
        else:
            print("error")
