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
loc = "/home/dal993192/dtwa_squeezing/results/LOCATION1/LOCATION2/YY/runVV/"
temp_save_loc = "/home/dal993192/scratch/LOCATION1/LOCATION2/YY/runVV/"

Jx = -1.0
Jy = -1.0
Jz = 2.0
hX = 0.0
hY = 0.0
hZ = 0.0
alpha = 3.0

# Simulation parameters
samples = 128  #samples per batch
batches = 50   #int(total_samples / samples)
total_samples = samples * batches
Jeff_t = CC / 4.0  #factor 4 for spin to pauli matrix conversion 
Jeff = DD 
timesteps = 500
rtol = 10**(-7)
atol = 10**(-10)
num_cores = -1
plot_ED = "False"
L = XX # square lattice length 
Ninit = L**2

disorder_seed = ZZ
np.random.seed(disorder_seed)

#now draw Poisson distribution with mean of lambda to see if spin is accepted
#lambda0 = 2.0
#lattice_NVC_count = scipy.stats.poisson.rvs(lambda0, size=Ninit)	

# draw from random distribution, probability 1 - p of NVC on a site
p = TT

rand_draws = rand.uniform(0,1,Ninit)
lattice_NVC_count = np.ones(Ninit,dtype=int)
# Arbitrary angle correlator/variance
maxNu = 10000
nu_indices = np.arange(0,maxNu)
nu = 2 * np.pi * nu_indices / maxNu 

for ii in range(0,Ninit):
    if rand_draws[ii] < p:
        lattice_NVC_count[ii] = 0

positions = copy.copy(lattice_NVC_count)

# Generate interaction matrices
J_mat, xx_filt, yy_filt = methods.gen_matrices_2D_pbc_vacancy(Ninit, alpha, positions)

Jx_mat = Jx * J_mat
Jy_mat = Jy * J_mat
Jz_mat = Jz * J_mat    

N = Jx_mat.shape[0]  # Number of spins actually generated 

hX_mat = hX * np.ones(N)
hY_mat = hY * np.ones(N)
hZ_mat = hZ * np.ones(N)

# time parameters
total_time = Jeff_t / Jeff
dt = total_time * (1.0 + 6.0 * p**2) / float(timesteps) # time step to save at 
timevec = dt * np.arange(0,timesteps+1) # vector of saved times

# Construct a unique ID based on simulation parameters
param_id = f"N{N}_alpha{alpha:.2f}_Jx{Jx:.2f}_Jy{Jy:.2f}_Jz{Jz:.2f}_hX{hX:.2f}_hY{hY:.2f}_hZ{hZ:.2f}_total_samples{total_samples}_timesteps{timesteps}_dt{dt:.3f}"
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


# Classical position of spin 
S_init = np.zeros([N,3])
S_init[:,0] = 1.0     # X position
S_init[:,1] = 0.0     # Y position
S_init[:,2] = 0.0     # Z position


Sx_mean =  np.zeros([timesteps+1])
Sy_mean =  np.zeros([timesteps+1])
Sz_mean =  np.zeros([timesteps+1])
Mxy_mean = np.zeros(timesteps+1)

Sx_M2 =  np.zeros([timesteps+1])
Sy_M2 =  np.zeros([timesteps+1])
Sz_M2 =  np.zeros([timesteps+1])
Mxy_M2 = np.zeros(timesteps+1)

nSx = 0
nSy = 0
nSz = 0
nMxy = 0

nCorrX = 0
nCorrY = 0
nCorrZ = 0
nCorrYZ = 0
n_mat_Var = 0

CorrX_mean =  np.zeros([timesteps+1])
CorrY_mean =  np.zeros([timesteps+1]) 
CorrZ_mean =  np.zeros([timesteps+1])
CorrYZ_mean = np.zeros([timesteps+1])
mat_Var_mean = np.zeros([maxNu,timesteps+1])

CorrX_M2 =  np.zeros(timesteps+1)
CorrY_M2 =  np.zeros(timesteps+1) 
CorrZ_M2 =  np.zeros(timesteps+1)
CorrYZ_M2 = np.zeros(timesteps+1)
mat_Var_M2 = np.zeros([maxNu,timesteps+1])


for bb in range(0,batches):

    print(bb)

    Parallel(n_jobs=num_cores)(delayed(methods.dtwa_sc)(S_init, bb, ss, samples, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat, temp_save_loc, rtol, atol) for ss in range(0,samples)) 

    if bb == batches - 1:
        os.remove(Jx_path)
        os.remove(Jy_path)
        os.remove(Jz_path)
        os.remove(hX_path)
        os.remove(hY_path)
        os.remove(hZ_path)

    # initialize matrices 
    Sx_av =  np.zeros(timesteps+1)
    Sy_av =  np.zeros(timesteps+1)
    Sz_av =  np.zeros(timesteps+1)
    Mxy_av = np.zeros(timesteps+1)

    CorrZ_av =  np.zeros(timesteps+1)
    CorrX_av =  np.zeros(timesteps+1)
    CorrY_av =  np.zeros(timesteps+1) 
    CorrYZ_av = np.zeros(timesteps+1)


    for ss in range(0,samples):
        sx_sample = np.load(temp_save_loc + "Sx_sample_" + str(ss) + ".npy")
        sy_sample = np.load(temp_save_loc + "Sy_sample_" + str(ss) + ".npy")
        sz_sample = np.load(temp_save_loc + "Sz_sample_" + str(ss) + ".npy")

        Sx_sum_t = np.sum(sx_sample,0)    
        Sy_sum_t = np.sum(sy_sample,0)    
        Sz_sum_t = np.sum(sz_sample,0)   

        Sx_av += 1.0/ samples * Sx_sum_t
        Sy_av += 1.0/ samples * Sy_sum_t
        Sz_av += 1.0/ samples * Sz_sum_t
        Mxy_av += 1.0/ samples * np.sqrt(np.sum(sx_sample,0)**2 + np.sum(sy_sample,0)**2)

        CorrZ_av  += 1.0 / samples * Sz_sum_t**2
        CorrX_av  += 1.0 / samples * Sx_sum_t**2
        CorrY_av  += 1.0 / samples * Sy_sum_t**2  
        CorrYZ_av += 1.0 / samples * Sy_sum_t * Sz_sum_t

        del sx_sample, sy_sample, sz_sample
        
        if bb == batches - 1:
            os.remove(temp_save_loc + "Sx_sample_" + str(ss) + ".npy")
            os.remove(temp_save_loc + "Sy_sample_" + str(ss) + ".npy")
            os.remove(temp_save_loc + "Sz_sample_" + str(ss) + ".npy")


    # Calculate the running mean and variance for magnetization and correlators
    nSx, Sx_mean, Sx_std, Sx_M2 = methods.update_stats(Sx_av, nSx, Sx_mean, Sx_M2) 
    nSy, Sy_mean, Sy_std, Sy_M2 = methods.update_stats(Sy_av, nSy, Sy_mean, Sy_M2) 
    nSz, Sz_mean, Sz_std, Sz_M2 = methods.update_stats(Sz_av, nSz, Sz_mean, Sz_M2) 
    nMxy, Mxy_mean, Mxy_std, Mxy_M2 = methods.update_stats(Mxy_av, nMxy, Mxy_mean, Mxy_M2) 

    nCorrZ, CorrZ_mean, CorrZ_std, CorrZ_M2 = methods.update_stats(CorrZ_av, nCorrZ, CorrZ_mean, CorrZ_M2) 
    nCorrX, CorrX_mean, CorrX_std, CorrX_M2 = methods.update_stats(CorrX_av, nCorrX, CorrX_mean, CorrX_M2) 
    nCorrY, CorrY_mean, CorrY_std, CorrY_M2 = methods.update_stats(CorrY_av, nCorrY, CorrY_mean, CorrY_M2) 
    nCorrYZ, CorrYZ_mean, CorrYZ_std, CorrYZ_M2 = methods.update_stats(CorrYZ_av, nCorrYZ, CorrYZ_mean, CorrYZ_M2) 

    # Construct the expectation values to subtract from correlators in variance calc. Note: average conducted before product.
    Sy_Sy_av = Sy_av**2
    Sz_Sz_av = Sz_av**2  
    Sy_Sz_av = Sy_av * Sz_av

    ### Construct Variance for all possible angles in YZ plane
    # Correlator
    mat_Var_av = np.einsum("u,t->ut", np.cos(nu)**2 , CorrY_av) + np.einsum("u,t->ut", np.sin(nu)**2, CorrZ_av) - np.einsum("u,t->ut", 2 * np.sin(nu) * np.cos(nu), CorrYZ_av) 
    #Subtract expectation values
    mat_Var_av += -(np.einsum("u,t->ut", np.cos(nu)**2 , Sy_Sy_av) + np.einsum("u,t->ut", np.sin(nu)**2, Sz_Sz_av) - np.einsum("u,t->ut", 2 * np.sin(nu) * np.cos(nu), Sy_Sz_av)) 

    # Calculate the running mean and variance of this
    n_mat_Var, mat_Var_mean, mat_Var_std, mat_Var_M2 = methods.update_stats(mat_Var_av, n_mat_Var, mat_Var_mean, mat_Var_M2) 


    endTime = datetime.now()
    print(str(endTime - startTime),'Batch run time')

Vmin_mean = np.zeros(np.size(timevec))
Vmin_std = np.zeros(np.size(timevec))

for tt in range(0,np.size(timevec)):

    Vmin_nu = mat_Var_mean[:,tt]
    Vmin_nu_std = mat_Var_std[:,tt]
    arg = np.argmin(Vmin_nu)
    Vmin_mean[tt] = Vmin_nu[arg]
    Vmin_std[tt] = Vmin_nu_std[arg]


np.save(loc + "posmat.npy",[xx_filt,yy_filt])
np.save(loc + "Jeff_mat.npy",np.sum(J_mat,1))
np.save(loc + "Sy.npy",Sy_mean)
np.save(loc + "Sx.npy",Sx_mean)
np.save(loc + "Sz.npy",Sz_mean)
np.save(loc + "Sy_std.npy",Sy_std)
np.save(loc + "Sx_std.npy",Sx_std)
np.save(loc + "Sz_std.npy",Sz_std)
np.save(loc + "Mxy.npy",Mxy_mean)
np.save(loc + "Mxy_std.npy",Mxy_std)
np.save(loc + "SySz.npy", CorrYZ_mean)
np.save(loc + "SySz_std.npy", CorrYZ_std)
np.save(loc + "SySy.npy",CorrY_mean)
np.save(loc + "SySy_std.npy",CorrY_std)
np.save(loc + "SxSx.npy",CorrX_mean)
np.save(loc + "SxSx_std.npy",CorrX_std)
np.save(loc + "SzSz.npy",CorrZ_mean)
np.save(loc + "SzSz_std.npy",CorrZ_std)
np.save(loc + "timevec.npy",timevec)
np.save(loc + "Vmin.npy",Vmin_mean)
np.save(loc + "Vmin_std.npy",Vmin_std)

endTime = datetime.now()
print(str(endTime - startTime),'Total run time')
