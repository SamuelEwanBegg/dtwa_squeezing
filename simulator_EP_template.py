import numpy as np
import matplotlib.pyplot as plt
import methods 
import numpy.random as rand
from joblib import Parallel, delayed, dump, load
import copy
import scipy.integrate
from datetime import datetime
startTime = datetime.now()

import os
print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK')}")
print(f"SLURM_JOB_CPUS_PER_NODE: {os.environ.get('SLURM_JOB_CPUS_PER_NODE')}")

print(str(startTime),'Commence code')

#save file location
loc = "/home/dal993192/dtwa_squeezing/results/LOCATION/YY/"
temp_save_loc = "/home/dal993192/scratch/LOCATION/YY/"

Jx_in = 1.0
Jy_in = 1.0
Jz_in = 1.0
Jx_out = 1.0
Jy_out = 1.0
Jz_out = 0.0
hX = 0.0
hY = 0.0
hZ = 0.0
alpha = 0.0
az = 10.0

# Simulation parameters
L = XX
Nval = L**2
samples = 640 #samples per batch
batches = 5   #int(total_samples / samples)
total_samples = samples * batches
timesteps = 200
rtol = 10**(-7)
atol = 10**(-10)
num_cores = -1
plot_ED = "False"


# Construct a unique ID based on simulation parameters
param_id = f"N{Nval}_alpha{alpha:.2f}_Jx_in{Jx_in:.2f}_Jy_in{Jy_in:.2f}_Jz_in{Jz_in:.2f}_Jx_out{Jx_out:.2f}_Jy_out{Jy_out:.2f}_Jz_out{Jz_out:.2f}_hX{hX:.2f}_hY{hY:.2f}_hZ{hZ:.2f}_total_samples{total_samples}"
print(param_id)

# Generate interaction matrices for in-plane interactions
Jx_mat_in = Jx_in * methods.gen_matrices_2D_pbc(Nval, alpha)
Jy_mat_in = Jy_in * methods.gen_matrices_2D_pbc(Nval, alpha)
Jz_mat_in = Jz_in * methods.gen_matrices_2D_pbc(Nval, alpha)

# Generate interaction matrices for out-plane interactions
Jx_mat_out = Jx_out * methods.gen_matrices_2D_pbc_bilayer(Nval, alpha, az)
Jy_mat_out = Jy_out * methods.gen_matrices_2D_pbc_bilayer(Nval, alpha, az)
Jz_mat_out = Jz_out * methods.gen_matrices_2D_pbc_bilayer(Nval, alpha, az)

Vavg = 1.0/Nval**2 * np.sum(Jx_mat_out)  
total_time = 15.0 / (Vavg * Nval) 
dt = total_time / float(timesteps)

Jx_mat = np.zeros([2*Nval, 2*Nval])
Jy_mat = np.zeros([2*Nval, 2*Nval])
Jz_mat = np.zeros([2*Nval, 2*Nval])

Jx_mat[0:Nval, 0:Nval] = 0.5 * Jx_mat_in
Jx_mat[Nval:, Nval:]   = 0.5 * Jx_mat_in
Jx_mat[0:Nval, Nval:]  = 0.5 * Jx_mat_out
Jx_mat[Nval:, 0:Nval]  = 0.5 * Jx_mat_out

Jy_mat[0:Nval, 0:Nval] = 0.5 * Jy_mat_in
Jy_mat[Nval:, Nval:]  =  0.5 * Jy_mat_in
Jy_mat[0:Nval, Nval:]  = 0.5 * Jy_mat_out
Jy_mat[Nval:, 0:Nval]  = 0.5 * Jy_mat_out

Jz_mat[0:Nval, 0:Nval] = 0.5 * Jz_mat_in
Jz_mat[Nval:, Nval:]   = 0.5 * Jz_mat_in
Jz_mat[0:Nval, Nval:]  = 0.5 * Jz_mat_out
Jz_mat[Nval:, 0:Nval]  = 0.5 * Jz_mat_out


hX_mat_upper = - hX * np.ones(Nval)
hY_mat_upper = - hY * np.ones(Nval)
hZ_mat_upper = - hZ * np.ones(Nval)

hX_mat_lower = hX * np.ones(Nval)
hY_mat_lower = hY * np.ones(Nval)
hZ_mat_lower = hZ * np.ones(Nval)

hX_mat = np.ones(2*Nval)
hY_mat = np.ones(2*Nval)
hZ_mat = np.ones(2*Nval)

hX_mat[0:Nval] = hX_mat_upper
hX_mat[Nval:] = hX_mat_lower

hY_mat[0:Nval] = hY_mat_upper
hY_mat[Nval:] = hY_mat_lower

hZ_mat[0:Nval] = hZ_mat_upper
hZ_mat[Nval:] = hZ_mat_lower


# File paths with unique suffix
Jx_path = f"{temp_save_loc}/pkl_store/Jx_mat_{param_id}.pkl"
Jy_path = f"{temp_save_loc}/pkl_store/Jy_mat_{param_id}.pkl"
Jz_path = f"{temp_save_loc}/pkl_store/Jz_mat_{param_id}.pkl"
hX_path = f"{temp_save_loc}/pkl_store/hX_mat_{param_id}.pkl"
hY_path = f"{temp_save_loc}/pkl_store/hY_mat_{param_id}.pkl"
hZ_path = f"{temp_save_loc}/pkl_store/hZ_mat_{param_id}.pkl"

# Save
def always_dump(obj, filename):
    dump(obj, filename)

always_dump(Jx_mat, Jx_path)
always_dump(Jy_mat, Jy_path)
always_dump(Jz_mat, Jz_path)
always_dump(hX_mat, hX_path)
always_dump(hY_mat, hY_path)
always_dump(hZ_mat, hZ_path)

# Load using mmap_mode for shared access
Jx_mat = load(Jx_path, mmap_mode='r')
Jy_mat = load(Jy_path, mmap_mode='r')
Jz_mat = load(Jz_path, mmap_mode='r')
hX_mat = load(hX_path, mmap_mode='r')
hY_mat = load(hY_path, mmap_mode='r')
hZ_mat = load(hZ_path, mmap_mode='r')

# define for later use
timevec = dt * np.arange(0,timesteps+1)

# Classical position of spin 
S_init = np.zeros([2*Nval,3])
#Upper layer
S_init[0:Nval,0] = 0.0    # X position
S_init[0:Nval,1] = 0.0    # Y position
S_init[0:Nval,2] = 1.0    # Z position

#Lower layer
S_init[Nval:,0] = 0.0     # X position
S_init[Nval:,1] = 0.0     # Y position
S_init[Nval:,2] = - 1.0   # Z position

CorrZ_mean_batch = []
CorrY_mean_batch = []
CorrX_mean_batch = []
CorrYZ_mean_batch = []
CorrXY_mean_batch = []

CorrZ_std_batch = []
CorrY_std_batch = []
CorrX_std_batch = []
CorrYZ_std_batch = []
CorrXY_std_batch = []

Sx_mean_batch = []
Sy_mean_batch = []
Sz_mean_batch = []
Mxy_mean_batch = []
Sx_std_batch = []
Sy_std_batch = []
Sz_std_batch = []
Mxy_std_batch = []


for bb in range(0,batches):

    print(bb)

    Parallel(n_jobs=num_cores)(delayed(methods.dtwa_sc_bilayer)(S_init, bb, ss, samples, timevec, 2 * Nval, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat, temp_save_loc, rtol, atol) for ss in range(0,samples)) 

    # initialize matrices 
    Sx_av =  np.zeros([2*Nval,timesteps+1])
    Sy_av =  np.zeros([2*Nval,timesteps+1])
    Sz_av =  np.zeros([2*Nval,timesteps+1])
    Mxy_av = np.zeros(timesteps+1)

    CorrZ_av =  np.zeros([2*Nval,2*Nval,timesteps+1])
    CorrX_av =  np.zeros([2*Nval,2*Nval,timesteps+1])
    CorrY_av =  np.zeros([2*Nval,2*Nval,timesteps+1]) 
    CorrYZ_av = np.zeros([2*Nval,2*Nval,timesteps+1])
    CorrXY_av = np.zeros([2*Nval,2*Nval,timesteps+1])

    for ss in range(0,samples):
        sx_sample = np.load(temp_save_loc + "Sx_sample_" + str(ss) + ".npy")
        sy_sample = np.load(temp_save_loc + "Sy_sample_" + str(ss) + ".npy")
        sz_sample = np.load(temp_save_loc + "Sz_sample_" + str(ss) + ".npy")

        Sx_av += 1.0/ samples * sx_sample
        Sy_av += 1.0/ samples * sy_sample
        Sz_av += 1.0/ samples * sz_sample
        Mxy_av += 1.0/ samples * np.sqrt(np.sum(sx_sample,0)**2 + np.sum(sy_sample,0)**2)

        CorrZ_av  += 1.0 / samples * np.einsum('nt,mt->nmt', sz_sample, sz_sample)
        CorrX_av  += 1.0 / samples * np.einsum('nt,mt->nmt', sx_sample, sx_sample)
        CorrY_av  += 1.0 / samples * np.einsum('nt,mt->nmt', sy_sample, sy_sample)
        CorrYZ_av += 1.0 / samples * np.einsum('nt,mt->nmt', sy_sample, sz_sample)
        CorrXY_av += 1.0 / samples * np.einsum('nt,mt->nmt', sx_sample, sy_sample)

    # Add to batch lists
    Sx_mean_batch += [Sx_av]
    Sy_mean_batch += [Sy_av]
    Sz_mean_batch += [Sz_av]
    Mxy_mean_batch += [Mxy_av]

    CorrZ_mean_batch +=  [CorrZ_av] 
    CorrX_mean_batch += [CorrX_av] 
    CorrY_mean_batch += [CorrY_av] 
    CorrYZ_mean_batch += [CorrYZ_av] 
    CorrXY_mean_batch += [CorrXY_av] 

# Calculate averages and std over batches

Sx_mean = np.mean(Sx_mean_batch,0)
Sy_mean = np.mean(Sy_mean_batch,0)
Sz_mean = np.mean(Sz_mean_batch,0)
Mxy_mean = np.mean(Mxy_mean_batch,0)

Sx_std = np.std(Sx_mean_batch,0)
Sy_std = np.std(Sy_mean_batch,0)
Sz_std = np.std(Sz_mean_batch,0)
Mxy_std = np.std(Mxy_mean_batch,0)

CorrZ_mean = np.mean(CorrZ_mean_batch,0)
CorrX_mean = np.mean(CorrX_mean_batch,0)
CorrY_mean = np.mean(CorrY_mean_batch,0)
CorrYZ_mean = np.mean(CorrYZ_mean_batch,0)
CorrXY_mean = np.mean(CorrXY_mean_batch,0)

CorrZ_std = np.std(CorrZ_mean_batch,0)
CorrX_std = np.std(CorrX_mean_batch,0)
CorrY_std = np.std(CorrY_mean_batch,0)
CorrYZ_std = np.std(CorrYZ_mean_batch,0)
CorrXY_std = np.std(CorrXY_mean_batch,0)

endTime = datetime.now()
print(str(endTime - startTime),'Run time')

# Calculate the minimum variance XX_A + YY_B + X_A Y_B + Y_B X_A
Omin = CorrX_mean[0:Nval,0:Nval,:] + CorrY_mean[Nval:,Nval:,:] - 2 * CorrXY_mean[0:Nval,Nval:,:]
Vmin_mean = np.sum(np.sum(Omin,0),0)
Ostd = CorrX_std[0:Nval,0:Nval,:] + CorrY_std[Nval:,Nval:,:] - 2 * CorrXY_std[0:Nval,Nval:,:]
Vmin_std = np.sum(np.sum(Ostd,0),0)

Signal = np.sum(Sz_mean[0:Nval,:] - Sz_mean[Nval:,:],0)

np.save(loc + "Sy.npy",Sy_mean)
np.save(loc + "Sx.npy",Sx_mean)
np.save(loc + "Sz.npy",Sz_mean)
np.save(loc + "Mxy.npy",Mxy_mean)
np.save(loc + "Sy_std.npy",Sy_std)
np.save(loc + "Sx_std.npy",Sx_std)
np.save(loc + "Sz_std.npy",Sz_std)
np.save(loc + "Mxy_std.npy",Mxy_std)
#np.save(loc + "SxSy.npy",np.sum(np.sum(CorrXY_mean,0),0))
#np.save(loc + "SySz.npy",np.sum(np.sum(CorrYZ_mean,0),0))
#np.save(loc + "SySy.npy",np.sum(np.sum(CorrY_mean,0),0))
#np.save(loc + "SxSx.npy",np.sum(np.sum(CorrX_mean,0),0))
#np.save(loc + "SzSz.npy",np.sum(np.sum(CorrZ_mean,0),0))
#np.save(loc + "SySz_std.npy",np.sum(np.sum(CorrYZ_std,0),0))
#np.save(loc + "SySy_std.npy",np.sum(np.sum(CorrY_std,0),0))
#np.save(loc + "SxSx_std.npy",np.sum(np.sum(CorrX_std,0),0))
#np.save(loc + "SzSz_std.npy",np.sum(np.sum(CorrZ_std,0),0))
np.save(loc + "timevec.npy",timevec)
np.save(loc + "Vmin.npy",Vmin_mean)
np.save(loc + "Vmin_std.npy",Vmin_std)
np.save(loc + "SzA_minus_SzB.npy",Signal)
np.save(loc + "Vavg.npy",Vavg)


