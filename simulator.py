import numpy as np
import matplotlib.pyplot as plt
import methods 
import numpy.random as rand
from joblib import Parallel, delayed
import copy
import scipy.integrate
from datetime import datetime
startTime = datetime.now()

import os
print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK')}")
print(f"SLURM_JOB_CPUS_PER_NODE: {os.environ.get('SLURM_JOB_CPUS_PER_NODE')}")

print(str(startTime),'Commence code')

loc = "scratch/test/"

Jx = -1.0
Jy = -1.0
Jz = 3.8
hX = 0.0
hY = 0.0
hZ = 0.0
alpha = 1.5

# Simulation parameters
N = 2500
samples = 64 #samples per batch
batches = 100  #int(total_samples / samples)
total_samples = samples * batches
timesteps = 200
dt = 0.005 # save times 
num_cores = -1
plot_ED = "False"

# Generate interaction matrices
Jx_mat = Jx * methods.gen_matrices(N, alpha)
Jy_mat = Jy * methods.gen_matrices(N, alpha)
Jz_mat = Jz * methods.gen_matrices(N, alpha)
hX_mat = hX * np.ones(N)
hY_mat = hY * np.ones(N)
hZ_mat = hZ * np.ones(N)

# define for later use
timevec = dt * np.arange(0,timesteps+1)


# Classical position of spin 
S_init = np.zeros([N,3])
S_init[:,0] = 1.0     # X position
S_init[:,1] = 0.0     # Y position
S_init[:,2] = 0.0     # Z position

CorrZ_mean_batch = []
CorrY_mean_batch = []
CorrX_mean_batch = []
CorrYZ_mean_batch = []

CorrZ_std_batch = []
CorrY_std_batch = []
CorrX_std_batch = []
CorrYZ_std_batch = []

Sx_mean_batch = []
Sy_mean_batch = []
Sz_mean_batch = []
Sx_std_batch = []
Sy_std_batch = []
Sz_std_batch = []


for bb in range(0,batches):

    print(bb)

    Sx_samples = []
    Sy_samples = []
    Sz_samples = []

    output = Parallel(n_jobs=num_cores)(delayed(methods.dtwa)(S_init, bb, ss, samples, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat) for ss in range(0,samples)) 

    for ss in range(0,samples):
        Sx_samples = Sx_samples + [output[ss][0][:,:]]
        Sy_samples = Sy_samples + [output[ss][1][:,:]]
        Sz_samples = Sz_samples + [output[ss][2][:,:]]

    # Magnetization
    Sx_av = np.mean(Sx_samples,0)
    Sy_av = np.mean(Sy_samples,0)
    Sz_av= np.mean(Sz_samples,0)

    Sx_fluct = np.std(Sx_samples,0)
    Sy_fluct = np.std(Sy_samples,0) 
    Sz_fluct = np.std(Sz_samples,0)

    # Correlations matrix is too big, so perform running sum.
    CorrZ_av = np.zeros([np.size(Sz_samples,1),np.size(Sz_samples,1),np.size(Sz_samples,2)])
    CorrX_av = np.zeros([np.size(Sz_samples,1),np.size(Sz_samples,1),np.size(Sz_samples,2)])
    CorrY_av = np.zeros([np.size(Sz_samples,1),np.size(Sz_samples,1),np.size(Sz_samples,2)])
    CorrYZ_av = np.zeros([np.size(Sz_samples,1),np.size(Sz_samples,1),np.size(Sz_samples,2)])

    for ss in range(0,samples):
        
        CorrZ_av  += 1.0 / samples * np.einsum('nt,mt->nmt', Sz_samples[ss], Sz_samples[ss])
        CorrX_av  += 1.0 / samples * np.einsum('nt,mt->nmt', Sx_samples[ss], Sx_samples[ss])
        CorrY_av  += 1.0 / samples * np.einsum('nt,mt->nmt', Sy_samples[ss], Sy_samples[ss])
        CorrYZ_av += 1.0 / samples * np.einsum('nt,mt->nmt', Sy_samples[ss], Sz_samples[ss])


    # Add to batch lists
    Sx_mean_batch += [Sx_av]
    Sy_mean_batch += [Sy_av]
    Sz_mean_batch += [Sz_av]
    Sx_std_batch += [Sx_fluct]
    Sy_std_batch += [Sy_fluct]
    Sz_std_batch += [Sz_fluct]

    CorrZ_mean_batch +=  [CorrZ_av] 
    CorrX_mean_batch += [CorrX_av] 
    CorrY_mean_batch += [CorrY_av] 
    CorrYZ_mean_batch += [CorrYZ_av] 


CorrZ_mean = np.mean(CorrZ_mean_batch,0)
CorrX_mean = np.mean(CorrX_mean_batch,0)
CorrY_mean = np.mean(CorrY_mean_batch,0)
CorrYZ_mean = np.mean(CorrYZ_mean_batch,0)

Sx_mean = np.mean(Sx_mean_batch,0)
Sy_mean = np.mean(Sy_mean_batch,0)
Sz_mean = np.mean(Sz_mean_batch,0)

Sx_std = np.std(Sx_mean_batch,0)
Sy_std = np.std(Sy_mean_batch,0)
Sz_std = np.std(Sz_mean_batch,0)

endTime = datetime.now()
print(str(endTime - startTime),'Run time')

# Arbitrary angle correlator/variance
maxNu = 100

Vmin_mean = np.zeros(np.size(timevec))
Vmin_std = np.zeros(np.size(timevec))

nu_indices = np.arange(0,maxNu)
nu = 2 * np.pi * nu_indices / maxNu 

temp_VMin = np.einsum("u,st->ust", np.cos(nu)**2 , np.sum(np.sum(CorrY_mean_batch,1),1)) + np.einsum("u,st->ust", np.sin(nu)**2, np.sum(np.sum(CorrZ_mean_batch,1),1)) - np.einsum("u,st->ust", 2 * np.sin(nu) * np.cos(nu), np.sum(np.sum(CorrYZ_mean_batch,1),1)) 

for tt in range(0,np.size(timevec)):

    Vmin_nu = np.mean(temp_VMin[:,:,tt],1)
    Vmin_nu_std = np.std(temp_VMin[:,:,tt],1)
    arg = np.argmin(Vmin_nu)
    Vmin_mean[tt] = Vmin_nu[arg]
    Vmin_std[tt] = Vmin_nu_std[arg]


np.save(loc + r"N="+str(N) + r"_samples=" +str(total_samples) + r"_dt="+str(dt)+ r"_Jz="+str(Jz)+"_Sy.npy",Sy_mean)
np.save(loc + r"N="+str(N) + r"_samples=" +str(total_samples) + r"_dt="+str(dt)+ r"_Jz="+str(Jz)+"_Sx.npy",Sx_mean)
np.save(loc + r"N="+str(N) + r"_samples=" +str(total_samples) + r"_dt="+str(dt)+ r"_Jz="+str(Jz)+"_Sz.npy",Sz_mean)
np.save(loc + r"N="+str(N) + r"_samples=" +str(total_samples) + r"_dt="+str(dt)+ r"_Jz="+str(Jz)+"_SySz.npy",np.sum(np.sum(CorrYZ_mean,0),0))
np.save(loc + r"N="+str(N) + r"_samples=" +str(total_samples) + r"_dt="+str(dt)+ r"_Jz="+str(Jz)+"_SySy.npy",np.sum(np.sum(CorrY_mean,0),0))
np.save(loc + r"N="+str(N) + r"_samples=" +str(total_samples) + r"_dt="+str(dt)+ r"_Jz="+str(Jz)+"_SxSx.npy",np.sum(np.sum(CorrX_mean,0),0))
np.save(loc + r"N="+str(N) + r"_samples=" +str(total_samples) + r"_dt="+str(dt)+ r"_Jz="+str(Jz)+"_timevec.npy",timevec)
np.save(loc + r"N="+str(N) + r"_samples=" +str(total_samples) + r"_dt="+str(dt)+ r"_Jz="+str(Jz)+"_Vmin.npy",Vmin_mean)
np.save(loc + r"N="+str(N) + r"_samples=" +str(total_samples) + r"_dt="+str(dt)+ r"_Jz="+str(Jz)+"_Vmin_std.npy",Vmin_std)
