import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import copy
from scipy.integrate import odeint
dtype=torch.float

# for the plots
plt.rc('xtick', labelsize=16) 
plt.rcParams.update({'font.size': 16})

# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)

def dfx(x,f):
    # Calculate the derivative with auto-differention
    return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]

def RollingAvg(a, m): #m-Average of vector a 
    n = len(a)
    Avg = np.diff(a[n-m:])
    Avg = np.sum(Avg*Avg)/(m-1)
    return Avg

def perturbPoints(grid, sig=0.03): #Peturbation of a given grid with fixed boundaries
    t0 = grid[0]
    tf = grid[-1]
    sig = sig*tf
#   stochastic perturbation of the evaluation points
#   force t[0]=t0  & force points to be in the t-interval
    delta_t = grid[1] - grid[0]  
    noise = delta_t * torch.randn_like(grid)*sig
    t = grid + noise
    t.data[t<t0]=t0 - t.data[t<t0]
    t.data[t>tf]=2*tf - t.data[t>tf]
    t.data[0] = torch.ones(1,1)*t0
    t.data[-1] = torch.ones(1,1)*tf
    t.requires_grad = False
    return t

def parametricSolutions(t, N1, t0, t1, fb):
    # parametric solutions 
    dt0 =t-t0
    dt1 = t- t1
    #f = (1-torch.exp(-dt0))*(1-torch.exp(-dt1)) -->Used on the paper
    f = torch.tanh(dt0)*torch.tanh(dt1) #-->Updated
    psi_hat  = fb  + f*N1
    return psi_hat


def hamEqs_Loss(t,psi, E): #Eigenvalue equation loss
    psi_dx = dfx(t,psi)
    psi_ddx= dfx(t,psi_dx)
    f = psi_ddx/2 -t*t*psi/2 + E*psi
    L  = (f.pow(2)).mean(); 
    return L

def L_norm(psi, x_L, x_R, M): #Normalisation loss
    f= (torch.squeeze(psi.T@psi) -M/(x_R-x_L)).pow(2)
    return f

def L_orth(psi, psi_eigen): #Orthogonality loss
    f = psi.T @ psi_eigen
    f = (torch.squeeze(f)).pow(2)
    return f

def Load_Eigens(dic, psi_eigen): #Optional: Load some known eigenvectors; continue searching for the rest
    XX = perturbPoints(torch.linspace(t0, tf, n_train).reshape(-1,1), sig=0)
    XX.requires_grad = True
    EvenSymmetry = True
    for state in range(0, len(dic)):
      mod = qNN1(neurons)
      mod.load_state_dict(dic[state])
      UU_PINN,_ = mod(XX,EvenSymmetry)
      UU_PINN = parametricSolutions(XX,UU_PINN, t0, tf,0)
      psi_eigen += UU_PINN.detach().numpy()
      EvenSymmetry = not EvenSymmetry
    return psi_eigen      
            
class qNN1(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(qNN1,self).__init__()

        # Define the Activation
        self.actF = mySin()

        # define layers
        #self.Lin_1   = torch.nn.Linear(1, D_hid)
        #self.E_out = torch.nn.Linear(D_hid, 1)
        #self.Lin_2 = torch.nn.Linear(D_hid, D_hid)
        #self.Ein = torch.nn.Linear(1,1)
        #self.Lin_out = torch.nn.Linear(D_hid+1, 1)
        
        self.Ein    = torch.nn.Linear(1,1)
        self.Lin_1  = torch.nn.Linear(2, 2*D_hid)
        self.Lin_2  = torch.nn.Linear(2*D_hid, D_hid)
        self.out    = torch.nn.Linear(D_hid, 1)

    def forward(self,t, EvenSymmetry): #Load both x, -x and export even/odd symmetric output
        In1 = self.Ein(torch.ones_like(t))
        
        L1 = self.Lin_1(torch.cat((t, In1),1))
        L1m = self.Lin_1(torch.cat((-t, In1),1))
        h1 = self.actF(L1)
        h1m = self.actF(L1m)

        L2 = self.Lin_2(h1)
        L2m = self.Lin_2(h1m)
        h2 = self.actF(L2)
        h2m = self.actF(L2m)
        out = self.out(h2)
        outm = self.out(h2m)

        if EvenSymmetry is True:
          Hub = 0.5*(out + outm)
        else:
          Hub = 0.5*(out - outm)

        return Hub, In1


def run_Scan_finitewell(dic, Loss_history, runTime, t0, tf, x1, neurons, epochs, n_train,lr, EvenSymmetry, minibatch_number = 1, p0=1e-2, L0= 1e-2): #9e-3
    model = qNN1(neurons)
    fc=None; 
    betas = [0.9, 0.99]
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    E_bin = len(dic)
    
    grid = torch.linspace(t0, tf, n_train).reshape(-1,1)
    
    ## TRAINING ITERATION    
    TeP0 = time.time()
    psi_eigen = torch.zeros(int(n_train/minibatch_number),1)
    psi_eigen = Load_Eigens(dic, psi_eigen) #In case we load some known eigenvectors
    

    for tt in range(epochs): 
        #adjusting learning rate at epoch 3 e3
        
        if tt % 3000 == 2999:
          for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 5
            
# Perturbing the evaluation points & forcing t[0]=t0
        t=perturbPoints(grid)
            
# BATCHING
        batch_size = int(n_train/minibatch_number)
        batch_start, batch_end = 0, batch_size
        

        #idx = np.random.permutation(n_train)
        #t_b = tb[idx] --> In case we want to shuffle the input vector
        t_b.requires_grad = True
        loss=0.0

        for nbatch in range(minibatch_number): 
# batch time set
            t_mb = t_b[batch_start:batch_end]

#  Network solutions 
            nn, En = model(t_mb, EvenSymmetry)

            psi  = parametricSolutions(t_mb, nn, t0, tf, x1) 

            L_DE = hamEqs_Loss(t_mb, psi, En) 
            L_Reg = L_norm(psi, t0, tf, batch_size) + L_orth(psi, psi_eigen)
            L_tot = L_DE+L_Reg

            Loss_history[1].append(En.mean().data.numpy()) 
            
# OPTIMIZER
            L_tot.backward(retain_graph=False); #True
            optimizer.step(); loss += L_tot.data.numpy()
            optimizer.zero_grad()

            batch_start += batch_size
            batch_end += batch_size

# keep the loss function history
        if tt % 100 == 99:
          print('epoch', tt + 1, ", loss: ", L_tot.item(), L_orth(psi, psi_eigen).item(), L_norm(psi, t0, tf, batch_size).item())
          

        Loss_history[0].append(loss)       
        patience = RollingAvg(Loss_history[0], 30)
#Keep the best model (low loss & patience) by using a deep copy
        if  patience < p0:
          if loss < L0:
            fc = copy.deepcopy(model)
            dic[E_bin]= fc.state_dict()
            E_bin += 1
            grid = torch.linspace(t0, tf, n_train).reshape(-1,1)
            XX = perturbPoints(grid, sig=0)
            XX.requires_grad = True
            UU_PINN,E = model(XX,EvenSymmetry)
            UU_PINN = parametricSolutions(XX,UU_PINN, t0, tf,0)
            V = UU_PINN.detach().numpy()
            
            plt.plot(grid, V)
            plt.plot(grid, np.pi**(-1/4)*np.exp(-grid*grid/2))
            plt.show()
            
            psi_eigen += psi.detach().numpy()
            
            print('Found')
            print(En.mean())
            E = E.mean()
            print('L DE', hamEqs_Loss(XX, UU_PINN, E) )
            print('Norm', L_norm(psi, t0, tf, batch_size))
            print('Orth', L_orth(psi, psi_eigen))
            EvenSymmetry = not EvenSymmetry
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr
              
              

    TePf = time.time()
    runTime += TePf - TeP0  
    return dic, Loss_history, runTime, EvenSymmetry



t0 = -6.
tf = 6.
xBC1=0.
dic ={}
loss_hists = ([], [])
runTime = 0
n_train, neurons, epochs, lr, EvenSymmetry = 60, 80, int(60e3), 8e-3, True
dic,loss_hists,runTime,EvenSymmetry = run_Scan_finitewell(dic,loss_hists,runTime, t0, tf, xBC1, neurons, epochs, n_train, lr, EvenSymmetry)


print('Training time (minutes):', runTime/60)
f = plt.figure(1)
plt.plot(loss_hists[0][::1000],'-b',alpha=0.975);
plt.yscale('log')

plt.tight_layout()
plt.ylabel('Total Loss');plt.xlabel('Epochs')
plt.show()
#plt.savefig(imgdir+'infinite_total_loss.png', bbox_inches = 'tight')
