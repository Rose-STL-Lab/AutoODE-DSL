import torch
import torch.nn as nn
import numpy as np
from ode_nn.torchdiffeq import odeint_adjoint as odeint
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######## Auto-FC ########
class Auto_FC(nn.Module):
    def __init__(self, input_length, input_dim, output_dim, hidden_dim):
        super(Auto_FC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.model = nn.Sequential(
            nn.Linear(input_length*input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, xx, output_length):
        xx = xx.reshape(xx.shape[0], -1)
        outputs = []
        for i in range(output_length):
            out = self.model(xx)
            xx = torch.cat([xx[:, self.input_dim:], out], dim = 1)  
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim = 1)

######## Seq2Seq ########
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate = 0):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout = dropout_rate, batch_first = True)
        
    def forward(self, source):
        outputs, hidden = self.lstm(source)
        return outputs, hidden
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout_rate = 0):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers = num_layers, 
                            dropout = dropout_rate, batch_first = True)
        
        self.out = nn.Linear(hidden_dim, output_dim)
      
    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)   
        prediction = self.out(output.float())
        return prediction, hidden     
    
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = num_layers).to(device)
        self.decoder = Decoder(output_dim = output_dim, hidden_dim = hidden_dim, num_layers = num_layers).to(device)
        self.output_dim = output_dim
        
    def forward(self, source, target_length):
        batch_size = source.size(0) 
        input_length = source.size(1) 
        output_dim = self.decoder.output_dim
        encoder_output, encoder_hidden = self.encoder(source)

        decoder_output = torch.zeros((batch_size, 1, output_dim), device = device)
        decoder_hidden = encoder_hidden
        
        outputs = []
        for t in range(target_length):  
            decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden)
            outputs.append(decoder_output)
        return torch.cat(outputs, dim = 1)   
    
    
######## Transformer ########
#Tranformer Encoder Only
class Transformer2(nn.Module):
    def __init__(self, input_dim, output_dim, nhead = 4, d_model = 128, num_layers = 6, dim_feedforward = 256):
        super(Transformer2, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout = 0)#.to(device)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout = 0)#.to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)#.to(device)
        #self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)#.to(device)
        self.embedding = nn.Linear(input_dim, d_model)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.output_dim = output_dim

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        #mask = mask.float().masked_fill(mask == 0, float('-inf'))#
        return mask
    
    def forward(self, xx, output_length, yy = None):
        src = self.embedding(xx).transpose(0,1)
        src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        encoder_output = self.transformer_encoder(src, mask = src_mask)#
        outputs = []
        for i in range(output_length):
            out = self.output_layer(encoder_output).transpose(0,1)[:,-1:]
            xx = torch.cat([xx[:,1:], out], dim = 1)
            outputs.append(out)
        return torch.cat(outputs, dim = 1)
 

#Tranformer Encoder-Decoder
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead = 4, d_model = 128, num_layers = 6, dim_feedforward = 256):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout = 0)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout = 0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)
        self.embedding = nn.Linear(input_dim, d_model)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.output_dim = output_dim

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    # teacher forcing: feed true yy during training
    # yy is None during inference
    def forward(self, xx, output_length, yy = None):
        src = self.embedding(xx)
        src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        encoder_output = self.transformer_encoder(src, mask = src_mask)
        decoder_output = []
        # greedy decode
        if yy is None:
            tgt = xx[:,-1:].transpose(0,1)
            for i in range(output_length):
                tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(device)
                out = self.transformer_decoder(self.embedding(tgt), encoder_output, tgt_mask = tgt_mask)
                tgt = torch.cat([tgt, self.output_layer(out[-1:])], dim = 0)
            out = tgt[1:].transpose(0,1)
        else:
            tgt_beg = xx[:,-1:].transpose(0,1)
            tgt = self.embedding(torch.cat([tgt_beg, yy.transpose(0,1)[:-1]], dim = 0))
            tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(device)
            out = self.transformer_decoder(tgt, encoder_output, tgt_mask = tgt_mask)
            out = self.output_layer(out).transpose(0,1)
        return out            
        
########### Latent ODE #########  
class Latent_ODE(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20, nbatch=1):
        super(Latent_ODE, self).__init__()
        self.func = LatentODEfunc(latent_dim, nhidden)#.to(device)
        self.rec = RecognitionRNN(latent_dim, obs_dim, nhidden)#.to(device)
        self.dec = LatentODEDecoder(latent_dim, obs_dim, nhidden)#.to(device)
        
    def forward(self, xx, output_length):
        time_steps = torch.linspace(0, 59, 60).float().to(device)[:output_length]
        out = self.rec.forward(torch.flip(xx, [1]))
        z0 = out
        pred_z = odeint(self.func, z0, time_steps).permute(1, 0, 2)
        pred_x = self.dec(pred_z)
        return pred_x
    
class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.ELU(inplace=True),
            nn.Linear(nhidden, latent_dim)
        )
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.model(x)
        return out
    
class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.model = nn.RNN(obs_dim, nhidden, batch_first = True)
        self.linear = nn.Linear(nhidden, latent_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.shape[0], self.nhidden).to(device)
        output, hn = self.model(x, h0)
        return self.linear(output[:,-1])
    
class LatentODEDecoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(LatentODEDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, nhidden),
            nn.ReLU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(inplace=True),
            nn.Linear(nhidden, obs_dim)
        )
        
    def forward(self, z):
        out = self.model(z)
        return out

    
    
##### Neural Encoder + ODE Solvers #########
class Neural_ODE(nn.Module):
    def __init__(self, input_dim, input_length, hidden_dim, solver = "Euler", encoder = "fc"):
        super(Neural_ODE, self).__init__()
        
        if encoder == "fc":
            self.encode_fc = nn.Sequential(
                nn.Linear(input_length*input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 3)
            )
        else:
            self.encode_lstm_1 = nn.LSTM(input_dim, hidden_dim, num_layers = 3, 
                                         bidirectional = True, batch_first = True)
            self.encode_lstm_2 = nn.Linear(hidden_dim*2, 3)
        
        self.step = torch.tensor(1)#0.5
        self.encoder = encoder
        self.solver = solver
        

    def f_S(self, S_n, I_n, E_n, beta): 
        return  -beta*S_n*I_n
    
    def f_E(self, S_n, I_n, E_n, beta, sigma): 
        return beta*S_n*I_n - sigma*E_n
    
    def f_I(self, I_n, E_n, sigma, gamma):
        return sigma*E_n - gamma*I_n
    #self.mu * self.sigma * E_n - self.gamma*I_n
    
    def f_R(self, I_n, gamma):
        return gamma*I_n
    
    # dt is included in the ks
    def RK4_update(self, f_n, k1, k2, k3, k4):
        return f_n + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4) * self.step
    
    def RK4(self, t, initial, beta, gamma, sigma = None):
        
        S_pred = [initial[:,0:1]]
        E_pred = [initial[:,1:2]]
        I_pred = [initial[:,2:3]]
        R_pred = [initial[:,3:4]]
        for n in range(len(t)-1):
            # dt * f(t[n], y[n]) 
            k1_S = self.f_S(S_pred[n], I_pred[n], E_pred[n], beta)
            k1_E = self.f_E(S_pred[n], I_pred[n], E_pred[n], beta, sigma)
            k1_I = self.f_I(I_pred[n], E_pred[n], sigma, gamma)
            k1_R = self.f_R(I_pred[n], gamma)
            
            # dt * f(t[n] + dt/2, y[n] + k1/2)
            S_plus_k1_half = S_pred[n] + k1_S / 2 * self.step
            I_plus_k1_half = I_pred[n] + k1_I / 2 * self.step
            E_plus_k1_half = E_pred[n] + k1_E / 2 * self.step
                       
            k2_S = self.f_S(S_plus_k1_half, I_plus_k1_half, E_plus_k1_half, beta)
            k2_E = self.f_E(S_plus_k1_half, I_plus_k1_half, E_plus_k1_half, beta, sigma)
            k2_I = self.f_I(I_plus_k1_half, E_plus_k1_half, sigma, gamma)
            k2_R = self.f_R(I_plus_k1_half, gamma)
            
            # dt * f(t[n] + dt/2, y[n] + k2/2)
            S_plus_k2_half = S_pred[n] + k2_S / 2 * self.step
            I_plus_k2_half = I_pred[n] + k2_I / 2 * self.step
            E_plus_k2_half = E_pred[n] + k2_E / 2 * self.step
            
            k3_S = self.f_S(S_plus_k2_half, I_plus_k2_half, E_plus_k2_half, beta)
            k3_E = self.f_E(S_plus_k2_half, I_plus_k2_half, E_plus_k2_half, beta, sigma)
            k3_I = self.f_I(I_plus_k2_half, E_plus_k2_half, sigma, gamma)
            k3_R = self.f_R(I_plus_k2_half, gamma)
            
            # dt * f(t[n] + dt, y[n] + k3) 
            S_plus_k3 = S_pred[n] + k3_S * self.step
            I_plus_k3 = I_pred[n] + k3_I * self.step
            E_plus_k3 = E_pred[n] + k3_E * self.step
            
            k4_S = self.f_S(S_plus_k3, I_plus_k3, E_plus_k3, beta)           
            k4_E = self.f_E(S_plus_k3, I_plus_k3, E_plus_k3, beta, sigma)
            k4_I = self.f_I(I_plus_k3, E_plus_k3, sigma, gamma) 
            k4_R = self.f_R(I_plus_k3, gamma)

            
            S_pred.append(self.RK4_update(S_pred[n], k1_S, k2_S, k3_S, k4_S))
            E_pred.append(self.RK4_update(E_pred[n], k1_E, k2_E, k3_E, k4_E))
            I_pred.append(self.RK4_update(I_pred[n], k1_I, k2_I, k3_I, k4_I))
            R_pred.append(self.RK4_update(R_pred[n], k1_R, k2_R, k3_R, k4_R))

        y_pred = torch.cat([torch.cat(S_pred, dim = 1).unsqueeze(-1), 
                            torch.cat(E_pred, dim = 1).unsqueeze(-1),
                            torch.cat(I_pred, dim = 1).unsqueeze(-1), 
                            torch.cat(R_pred, dim = 1).unsqueeze(-1)], dim = 2)
        return y_pred
            
    
            
    def forward(self, xx, output_length):
        if self.encoder == "fc":
            out = self.encode_fc(xx.reshape(xx.shape[0], -1))
        elif self.encoder == "lstm":
            out =  self.encode_lstm_2(self.encode_lstm_1(xx)[0][:,-1])
        else:
            return "Error"
            
            
        t = torch.linspace(0, 60//2, 61).float().cuda()[:output_length]
        
        if self.solver == "Euler":
            return self.Euler(t, xx[:,0], out[:,0:1], out[:,1:2],  out[:,2:3])
        elif self.solver == "RK4":
            return self.RK4(t, xx[:,0], out[:,0:1], out[:,1:2],  out[:,2:3])
        else:
            return "Error" 