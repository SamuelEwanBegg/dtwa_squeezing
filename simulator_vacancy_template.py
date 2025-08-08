import numpy as np
import matplotlib.pyplot as plt
import methods 
import numpy.random as rand
from joblib import Parallel, delayed, dump, load
import copy
import scipy.integrate
import scipy.stats
from datetime import datetime
startTime = datetime.now()

import os
print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK')}")
print(f"SLURM_JOB_CPUS_PER_NODE: {os.environ.get('SLURM_JOB_CPUS_PER_NODE')}")

print(str(startTime),'Commence code')

#save file location
loc = "/home/dal993192/dtwa_squeezing/results/LOCATION/YY/runVV/"
temp_save_loc = "/home/dal993192/scratch/LOCATION/YY/runVV/"

Jx = -1.0
Jy = -1.0
Jz = 1.0
hX = 0.0
hY = 0.0
hZ = 0.0
alpha = 3.0

# Simulation parameters
samples = 640 #samples per batch
batches = 10   #int(total_samples / samples)
total_samples = samples * batches
timesteps = 65
dt = 0.005 # save times 
rtol = 10**(-7)
atol = 10**(-10)
num_cores = -1
plot_ED = "False"
L = PP # square lattice length 
Ninit = L**2
lambda0 = 1

disorder_seed = ZZ
np.random.seed(disorder_seed)

#now draw Poisson distribution with mean of lambda to see if spin is accepted
lattice_NVC_count = scipy.stats.poisson.rvs(lambda0, size=Ninit)	
positions = copy.copy(lattice_NVC_count)
positions[positions > 1] = 0 #set all non-zero values to 1

# Generate interaction matrices
Jx_mat, xx_filt, yy_filt, rmat = methods.gen_matrices_2D_pbc_vacancy(Ninit, alpha, positions)
Jy_mat, xx_filt, yy_filt, rmat = methods.gen_matrices_2D_pbc_vacancy(Ninit, alpha, positions)
Jz_mat, xx_filt, yy_filt, rmat = methods.gen_matrices_2D_pbc_vacancy(Ninit, alpha, positions)

Jx_mat = Jx * Jx_mat
Jy_mat = Jy * Jy_mat
Jz_mat = Jz * Jz_mat    

N = Jx_mat.shape[0]  # Number of spins actually generated 

hX_mat = hX * np.ones(N)
hY_mat = hY * np.ones(N)
hZ_mat = hZ * np.ones(N)

# Construct a unique ID based on simulation parameters
param_id = f"N{N}_alpha{alpha:.2f}_Jx{Jx:.2f}_Jy{Jy:.2f}_Jz{Jz:.2f}_hX{hX:.2f}_hY{hY:.2f}_hZ{hZ:.2f}_total_samples{total_samples}"
print(param_id)

# File paths with unique suffix
Jx_path = f"{temp_save_loc}/pkl_store/Jx_mat_{param_id}.pkl"
Jy_path = f"{temp_save_loc}/pkl_store/Jy_mat_{param_id}.pkl"
Jz_path = f"{temp_save_loc}/pkl_store/Jz_mat_{param_id}.pkl"
hX_path = f"{temp_save_loc}/pkl_store/hX_mat_{param_id}.pkl"
hY_path = f"{temp_save_loc}/pkl_store/hY_mat_{param_id}.pkl"
hZ_path = f"{temp_save_loc}/pkl_store/hZ_mat_{param_id}.pkl"

# Save only if they don't already exist
def maybe_dump(obj, filename):
    if not os.path.exists(filename):
        dump(obj, filename)

maybe_dump(Jx_mat, Jx_path)
maybe_dump(Jy_mat, Jy_path)
maybe_dump(Jz_mat, Jz_path)
maybe_dump(hX_mat, hX_path)
maybe_dump(hY_mat, hY_path)
maybe_dump(hZ_mat, hZ_path)

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
Mxy_mean_batch = []
Sx_std_batch = []
Sy_std_batch = []
Sz_std_batch = []
Mxy_std_batch = []


for bb in range(0,batches):

    print(bb)

    Parallel(n_jobs=num_cores)(delayed(methods.dtwa_sc)(S_init, bb, ss, samples, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat, temp_save_loc, rtol, atol) for ss in range(0,samples)) 

    # initialize matrices 
    Sx_av =  np.zeros([N,timesteps+1])
    Sy_av =  np.zeros([N,timesteps+1])
    Sz_av =  np.zeros([N,timesteps+1])
    Mxy_av = np.zeros(timesteps+1)

    CorrZ_av =  np.zeros([N,N,timesteps+1])
    CorrX_av =  np.zeros([N,N,timesteps+1])
    CorrY_av =  np.zeros([N,N,timesteps+1]) 
    CorrYZ_av = np.zeros([N,N,timesteps+1])


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

    # Add to batch lists
    Sx_mean_batch += [Sx_av]
    Sy_mean_batch += [Sy_av]
    Sz_mean_batch += [Sz_av]
    Mxy_mean_batch += [Mxy_av]

    CorrZ_mean_batch +=  [CorrZ_av] 
    CorrX_mean_batch += [CorrX_av] 
    CorrY_mean_batch += [CorrY_av] 
    CorrYZ_mean_batch += [CorrYZ_av] 


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

CorrZ_std = np.std(CorrZ_mean_batch,0)
CorrX_std = np.std(CorrX_mean_batch,0)
CorrY_std = np.std(CorrY_mean_batch,0)
CorrYZ_std = np.std(CorrYZ_mean_batch,0)

endTime = datetime.now()
print(str(endTime - startTime),'Run time')

# Arbitrary angle correlator/variance
maxNu = 10000

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


np.save(loc + "posmat.npy",[xx_filt,yy_filt])
np.save(loc + "rmat.npy",rmat)
np.save(loc + "Sy.npy",Sy_mean)
np.save(loc + "Sx.npy",Sx_mean)
np.save(loc + "Sz.npy",Sz_mean)
np.save(loc + "Sy_std.npy",Sy_std)
np.save(loc + "Sx_std.npy",Sx_std)
np.save(loc + "Sz_std.npy",Sz_std)
np.save(loc + "Mxy.npy",Mxy_mean)
np.save(loc + "Mxy_std.npy",Mxy_std)
np.save(loc + "SySz.npy",np.sum(np.sum(CorrYZ_mean,0),0))
np.save(loc + "SySz_std.npy",np.sum(np.sum(CorrYZ_std,0),0))
np.save(loc + "SySy.npy",np.sum(np.sum(CorrY_mean,0),0))
np.save(loc + "SySy_std.npy",np.sum(np.sum(CorrY_std,0),0))
np.save(loc + "SxSx.npy",np.sum(np.sum(CorrX_mean,0),0))
np.save(loc + "SxSx_std.npy",np.sum(np.sum(CorrX_std,0),0))
np.save(loc + "SzSz.npy",np.sum(np.sum(CorrZ_mean,0),0))
np.save(loc + "SzSz_std.npy",np.sum(np.sum(CorrZ_std,0),0))
np.save(loc + "timevec.npy",timevec)
np.save(loc + "Vmin.npy",Vmin_mean)
np.save(loc + "Vmin_std.npy",Vmin_std)
