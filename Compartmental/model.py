import os
import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def weight_fun(num_steps, function = "linear", feat_weight = False):    
    if function == "linear":
        weight = torch.linspace(0, 1, num_steps).to(device).reshape(1,-1,1)*2/num_steps
    if function == "sqrt":
        sqrt_func = lambda x: torch.sqrt(x)
        weight = sqrt_func(torch.linspace(0, 1, num_steps).reshape(1,-1,1).to(device))/torch.sum(sqrt_func(torch.linspace(0, 1, num_steps).to(device)))
    if feat_weight:
        weight = weight.repeat(1,1,3)
        weight[:,:,2] =  weight[:,:,2] #* 12
        weight[:,:,1] =  weight[:,:,1] #* 4
    return weight

class SuEIR(nn.Module):
    def __init__(self, initial_I, initial_R, num_regions, solver = "Euler", low_rank = None):
        super(SuEIR, self).__init__()
        
        self.init_S = nn.Parameter(torch.rand(num_regions).to(device)/100)
        self.init_E = nn.Parameter(torch.rand(num_regions).to(device)/100)
        self.init_I = initial_I
        self.init_R = initial_R
        if num_regions == 1:
            self.A = torch.ones(1, 1).to(device)
        else:
            if low_rank:
                self.B = nn.Parameter(torch.rand(num_regions, low_rank).to(device)/100)
                self.C = nn.Parameter(torch.rand(low_rank, num_regions).to(device)/100)
                self.A = torch.mm(self.B, self.C)
            else:
                self.A = nn.Parameter(torch.rand(num_regions, num_regions).to(device)/100)
        
        self.beta = nn.Parameter(torch.rand(num_regions).to(device)/100)
        self.gamma = nn.Parameter(torch.rand(num_regions).to(device)/100)
        self.sigma = nn.Parameter(torch.rand(num_regions).to(device)/100)
        self.mu = nn.Parameter(torch.rand(num_regions).to(device)/100)
        self.step = nn.Parameter(torch.rand(1).to(device)/100)
        self.solver = solver
        
    def Euler(self, num_steps):
        S_pred = [self.init_S]
        E_pred = [self.init_E]
        I_pred = [self.init_I]
        R_pred = [self.init_R]
        for n in range(num_steps-1):
            S_pred.append(S_pred[n] - self.beta * (torch.mm(self.A, ((I_pred[n] + E_pred[n]) * S_pred[n]).reshape(-1,1)).squeeze(1)) * self.step)
            E_pred.append(E_pred[n] + (self.beta * S_pred[n] * (I_pred[n]+ E_pred[n]) - self.sigma * E_pred[n]) * self.step)
            I_pred.append(I_pred[n] + (self.mu * self.sigma * E_pred[n] - self.gamma*I_pred[n]) * self.step)
            R_pred.append(R_pred[n] + self.gamma * I_pred[n] * self.step)

        y_pred = torch.cat([torch.stack(I_pred).transpose(0,1).unsqueeze(-1),
                            torch.stack(R_pred).transpose(0,1).unsqueeze(-1)], dim = -1)
        return y_pred
    
    def f_S(self, S_n, I_n, E_n): # TODO: Missing division by N =  I + R + S + E
        return -self.beta * (torch.mm(self.A, ((I_n + E_n) * S_n).reshape(-1,1)).squeeze(1))
    
    def f_E(self, S_n, I_n, E_n): # TODO: Missing diving self.beta * S_n * (I_n + E_n) by N
        return self.beta * S_n * (I_n + E_n) - self.sigma * E_n
    
    def f_I(self, I_n, E_n):
        return self.mu * self.sigma * E_n - self.gamma*I_n
    
    def f_R(self, I_n):
        return self.gamma*I_n
    
    # dt is included in the ks
    def RK4_update(self, f_n, k1, k2, k3, k4):
        return f_n + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4) * self.step
    
    def RK4(self, num_steps):
        S_pred = [self.init_S]
        E_pred = [self.init_E]
        I_pred = [self.init_I]
        R_pred = [self.init_R]
        for n in range(num_steps-1):
            # dt * f(t[n], y[n]) 
            k1_S = self.f_S(S_pred[n], I_pred[n], E_pred[n])
            k1_E = self.f_E(S_pred[n], I_pred[n], E_pred[n])
            k1_I = self.f_I(I_pred[n], E_pred[n])
            k1_R = self.f_R(I_pred[n])
            
            # dt * f(t[n] + dt/2, y[n] + k1/2)
            S_plus_k1_half = S_pred[n] + k1_S / 2 * self.step
            I_plus_k1_half = I_pred[n] + k1_I / 2 * self.step
            E_plus_k1_half = E_pred[n] + k1_E / 2 * self.step
            
            
            k2_S = self.f_S(S_plus_k1_half, I_plus_k1_half, E_plus_k1_half)
            k2_E = self.f_E(S_plus_k1_half, I_plus_k1_half, E_plus_k1_half)
            k2_I = self.f_I(I_plus_k1_half, E_plus_k1_half)
            k2_R = self.f_R(I_plus_k1_half)
            
            # dt * f(t[n] + dt/2, y[n] + k2/2)
            S_plus_k2_half = S_pred[n] + k2_S / 2 * self.step
            I_plus_k2_half = I_pred[n] + k2_I / 2 * self.step
            E_plus_k2_half = E_pred[n] + k2_E / 2 * self.step
            
            
            k3_S = self.f_S(S_plus_k2_half, I_plus_k2_half, E_plus_k2_half)
            k3_E = self.f_E(S_plus_k2_half, I_plus_k2_half, E_plus_k2_half)
            k3_I = self.f_I(I_plus_k2_half, E_plus_k2_half)
            k3_R = self.f_R(I_plus_k2_half)
            
            
            # dt * f(t[n] + dt, y[n] + k3) 
            S_plus_k3 = S_pred[n] + k3_S * self.step
            I_plus_k3 = I_pred[n] + k3_I * self.step
            E_plus_k3 = E_pred[n] + k3_E * self.step
            
            k4_S = self.f_S(S_plus_k3, I_plus_k3, E_plus_k3)           
            k4_E = self.f_E(S_plus_k3, I_plus_k3, E_plus_k3)
            k4_I = self.f_I(I_plus_k3, E_plus_k3) 
            k4_R = self.f_R(I_plus_k3)
            #print(I_pred[-1])
            S_pred.append(self.RK4_update(S_pred[n], k1_S, k2_S, k3_S, k4_S))
            E_pred.append(self.RK4_update(E_pred[n], k1_E, k2_E, k3_E, k4_E))
            I_pred.append(self.RK4_update(I_pred[n], k1_I, k2_I, k3_I, k4_I))
            R_pred.append(self.RK4_update(R_pred[n], k1_R, k2_R, k3_R, k4_R))
        y_pred = torch.cat([torch.stack(I_pred).transpose(0,1).unsqueeze(-1),
                            torch.stack(R_pred).transpose(0,1).unsqueeze(-1)], dim = -1)
        return y_pred
            
            
    def forward(self, num_steps):
        if self.solver == "Euler":
            return self.Euler(num_steps)
        elif self.solver == "RK4":
            return self.RK4(num_steps)
        else:
            print("Error")   
            
            
# class SuEIR_Corr(nn.Module):
#     def __init__(self, init_S, init_E, init_I, init_R, num_regions, step_size):
#         super(SuEIR_Corr, self).__init__()
        
#         self.init_S = init_S.to(device)
#         self.init_E = init_E.to(device)
#         self.init_I = init_I.to(device)
#         self.init_R = init_R.to(device)
#         self.A = torch.diag(torch.ones(10)).to(device)
        
#         self.beta = torch.linspace(0.3, 0.75, 10).to(device)
#         self.gamma = torch.tensor(0.2).repeat(num_regions).to(device)
#         self.sigma = torch.tensor(0.5).repeat(num_regions).to(device)
#         self.mu = torch.tensor(0.5).repeat(num_regions).to(device)
#         self.step = torch.tensor(step_size).to(device)
           
#     def f_S(self, S_n, I_n, E_n): # TODO: Missing division by N =  I + R + S + E
#         #print(self.A.shape, I_n.shape, E_n.shape, S_n.shape)
#         return -self.beta * (torch.mm(self.A, ((I_n + E_n) * S_n).reshape(-1,1)).squeeze(1))
    
#     def f_E(self, S_n, I_n, E_n): # TODO: Missing diving self.beta * S_n * (I_n + E_n) by N
#         return self.beta * S_n * (I_n + E_n) - self.sigma * E_n
    
#     def f_I(self, I_n, E_n):
#         return self.mu * self.sigma * E_n - self.gamma*I_n
    
#     def f_R(self, I_n):
#         return self.gamma*I_n
    
#     # dt is included in the ks
#     def RK4_update(self, f_n, k1, k2, k3, k4):
#         return f_n + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4) * self.step
    
#     def RK4(self, num_steps):
#         S_pred = [self.init_S]
#         E_pred = [self.init_E]
#         I_pred = [self.init_I]
#         R_pred = [self.init_R]
#         for n in range(num_steps-1):
#             # dt * f(t[n], y[n]) 
#             k1_S = self.f_S(S_pred[n], I_pred[n], E_pred[n])
#             k1_E = self.f_E(S_pred[n], I_pred[n], E_pred[n])
#             k1_I = self.f_I(I_pred[n], E_pred[n])
#             k1_R = self.f_R(I_pred[n])
            
#             # dt * f(t[n] + dt/2, y[n] + k1/2)
#             S_plus_k1_half = S_pred[n] + k1_S / 2 * self.step
#             I_plus_k1_half = I_pred[n] + k1_I / 2 * self.step
#             E_plus_k1_half = E_pred[n] + k1_E / 2 * self.step
            
            
#             k2_S = self.f_S(S_plus_k1_half, I_plus_k1_half, E_plus_k1_half)
#             k2_E = self.f_E(S_plus_k1_half, I_plus_k1_half, E_plus_k1_half)
#             k2_I = self.f_I(I_plus_k1_half, E_plus_k1_half)
#             k2_R = self.f_R(I_plus_k1_half)
            
#             # dt * f(t[n] + dt/2, y[n] + k2/2)
#             S_plus_k2_half = S_pred[n] + k2_S / 2 * self.step
#             I_plus_k2_half = I_pred[n] + k2_I / 2 * self.step
#             E_plus_k2_half = E_pred[n] + k2_E / 2 * self.step
            
            
#             k3_S = self.f_S(S_plus_k2_half, I_plus_k2_half, E_plus_k2_half)
#             k3_E = self.f_E(S_plus_k2_half, I_plus_k2_half, E_plus_k2_half)
#             k3_I = self.f_I(I_plus_k2_half, E_plus_k2_half)
#             k3_R = self.f_R(I_plus_k2_half)
            
            
#             # dt * f(t[n] + dt, y[n] + k3) 
#             S_plus_k3 = S_pred[n] + k3_S * self.step
#             I_plus_k3 = I_pred[n] + k3_I * self.step
#             E_plus_k3 = E_pred[n] + k3_E * self.step
            
#             k4_S = self.f_S(S_plus_k3, I_plus_k3, E_plus_k3)           
#             k4_E = self.f_E(S_plus_k3, I_plus_k3, E_plus_k3)
#             k4_I = self.f_I(I_plus_k3, E_plus_k3) 
#             k4_R = self.f_R(I_plus_k3)
            
#             S_pred.append(self.RK4_update(S_pred[n], k1_S, k2_S, k3_S, k4_S))
#             E_pred.append(self.RK4_update(E_pred[n], k1_E, k2_E, k3_E, k4_E))
#             I_pred.append(self.RK4_update(I_pred[n], k1_I, k2_I, k3_I, k4_I))
#             R_pred.append(self.RK4_update(R_pred[n], k1_R, k2_R, k3_R, k4_R))
            
#         y_pred = torch.cat([torch.stack(I_pred).transpose(0,1).unsqueeze(-1),
#                             torch.stack(R_pred).transpose(0,1).unsqueeze(-1)], dim = -1)
#         return y_pred
            
            
#     def forward(self, num_steps):
#         return self.RK4(num_steps)