import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm_notebook as tqdm
import math
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralLV():
    def __init__(self, num_time_series, num_time_steps, low_rank_param, 
                 is_full_matrix, p0, r, k, A, is_sym):
        # Define number of time series
        self.num_time_series = num_time_series
        # Define number of discrete time steps will assume the same for each time series so p_i(t) in R^(dxN)
        self.num_time_steps = num_time_steps
        self.low_rank_param = low_rank_param
        self.is_full_matrix = is_full_matrix
        self.p0 = p0
        self.r = r
        self.k = k
        k = nn.Parameter(torch.ones(num_time_series).float().cuda())
        self.A = A
        self.is_sym = is_sym
    
    # mat1 = A is the default case.  If low_rank is on, we have the symmetric case where A=B^TB 
    # and mat1 = B and in the non-symmetric case we need to pass in mat2 = C so A = B^TC
    def solve_discrete_lv(self, mat1, mat2=None, is_full_matrix=True):
        p = [] # need to store as list for autograd won't let you append indices in same matrix
        p.append(self.p0)
        for n in range(self.num_time_steps-1): # element-wise vector division and multiplication
            # Compute Ap to generate synthetic data for the full rank matrix A
            if is_full_matrix:
                mat_vec_prod = torch.mm(mat1, p[n].reshape(-1, 1)).squeeze(-1)
            else:
                mat_vec_prod = compute_mat_vec_prod(mat1, mat2, p[n].reshape(-1, 1)).squeeze(-1)
            p.append((1 + self.r * (1 - mat_vec_prod / self.k)*0.5) * p[n])
            # concat puts in nd array of size num_ts*N
            # need to take size (N, num_ts) and transpose otherwise default is doing row major
            # and we need column major storing of the linear list p
        return torch.cat(p, dim=0).reshape(self.num_time_steps, self.num_time_series).T
    
    """
    p0: initial condition shape (num_ts, )
    r: growth rate shape (num_ts, ) TODO: Learn
    k: carrying capacity shape (num_ts, ) TODO: Learn
    A: interaction matrix shape (num_ts, num_ts)
    """
    def run(self, num_epochs=1000, model=None):
        
        # Compute exact solution with full rank matrix
        p = self.solve_discrete_lv(self.A)
        # To initialize model otherwise can feed in model and rerun
        if model is None:
            model = LowRankVectorEmbedding(self).to(device)
            
#             model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, \
#                                           ctx=self.ctx)
        optimizer = torch.optim.Adam(model.parameters(), 0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.95)
        p_approx = train(p, optimizer, model, num_epochs)
        B = model.embedding_low_rank_mat_B(model.feat_static_cat).T
        C = None if self.is_sym else model.embedding_low_rank_mat_C(model.feat_static_cat).T
        A_approx = compute_low_rank_product(B, C)
        return p_approx, p, A_approx, model  

class LowRankVectorEmbedding(nn.Module):
    def __init__(self, neural_lv):
        super(LowRankVectorEmbedding, self).__init__()
        self.num_time_series = neural_lv.num_time_series # num_ts
        self.low_rank_param = neural_lv.low_rank_param # low rank parameter k
        self.is_full_matrix = neural_lv.is_full_matrix
        self.feat_static_cat = torch.arange(self.num_time_series).to(device)
        self.is_sym = neural_lv.is_sym
        self.neural_lv = neural_lv        
      
        self.embedding_low_rank_mat_B = nn.Embedding(self.num_time_series, self.low_rank_param)
        self.embedding_low_rank_mat_B.weight.data.uniform_(-0.1, 0.1)
        if not self.is_sym:
            self.embedding_low_rank_mat_C = nn.Embedding(self.num_time_series, self.low_rank_param)
            self.embedding_low_rank_mat_C.weight.data.uniform_(-0.1, 0.1)
    def forward(self):
        # find low rank vector computed per time series
        # feat_static_cat consists of the time series indices 0, ..., cardinality - 1
        # embedding returns (num_ts, low_rank_param) need to transpose it
        B = self.embedding_low_rank_mat_B(self.feat_static_cat).T
        C = None if self.is_sym else self.embedding_low_rank_mat_C(self.feat_static_cat).T
        # Explicitly form matrix matrix product A = B^T* CO(kd^2) expensive
        if self.is_full_matrix:
            return self.neural_lv.solve_discrete_lv(compute_low_rank_product(B, C))
        # Compute matrix vector product B^T * (C p) O(kd)
        else:
            return self.neural_lv.solve_discrete_lv(B, C, self.is_full_matrix)



def lv_plot_ts(p, p_approx, max_num_plots=10, num_rows=2, fig_size_width=10): # plots all time series at corresponding time point
    plt.rcParams["figure.figsize"] = (fig_size_width, 5)
    num_ts = p.shape[0]
    N = p.shape[1]
    t = np.arange(N)
    num_plots = min(num_ts, max_num_plots)
    num_cols = int(num_plots / num_rows)
    fig, axs = plt.subplots(num_rows, num_cols)
    for ts_idx in range(num_plots):
        plt.subplot(num_rows, num_cols, ts_idx+1) 
        plt.plot(t, p[ts_idx, :].cpu().data.numpy(), \
                 t, p_approx[ts_idx, :].cpu().data.numpy(), 'r--')
        plt.ylabel(f'$p_{ts_idx}(t)$')
        plt.xlabel('t')
        plt.legend(('Exact', 'Approx'))
        plt.xlabel('time: $t$')
        plt.ylabel(f'$p_{ts_idx}(t)$')


# Returns random samples fromuniform distribution [0,1] can change to randn for normally distributed random values
def generate_data(num_ts, seed=100): # num_ts = d
    #torch.manual_seed(seed)
    # vector of shape (num_ts, )
    r = torch.rand(num_ts,).float().to(device)
    # vector of shape (num_ts, )
    k = torch.rand(num_ts,).float().to(device)
    # matrix of shape (num_ts, num_ts)
    # diagonal entries are 1 representing intraspecies competition
    A = torch.rand(num_ts, num_ts).fill_diagonal_(1).float().to(device)
    # initial condition vector of shape (num_ts, )
    p0 = torch.rand(num_ts,).float().to(device)
    return r, k, p0, A

# Compute B^Tz, where z = C*p, C = B in the symmetic case
def compute_mat_vec_prod(B, C, p):
    # Can replace dot with nd.linalg.gemm2 for the matrix vector multiplication
    z = torch.mm(C, p) if C is not None else torch.mm(B, p)
    return torch.mm(B.transpose(0,1), z)


# A = B^TC, where C = B in the symmetric case
def compute_low_rank_product(B, C):
    return torch.mm(B.transpose(0,1), C) if C is not None else torch.mm(B.transpose(0,1), B)

def train(p, optimizer, model, num_epochs=1000):
    loss_fun = torch.nn.MSELoss()
    tqdm_epochs = tqdm(range(num_epochs))
    for e in tqdm_epochs:
        p_approx = model.forward()
        Loss = loss_fun(p, p_approx)
        optimizer.zero_grad()
        Loss.backward(retain_graph=True)
        optimizer.step()
        tqdm_epochs.set_postfix({'loss': Loss.item()})
    return p_approx